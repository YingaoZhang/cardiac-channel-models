from app.services.rag_service import summarize_herg_context
from app.services.molecular_review import review_candidate
from app.services.sa_score import calculate_sa_score, describe_sa_score
from app.state import AIDDState


SYNTHESIS_SA_THRESHOLD = 7.0


def _format_prob(value):
    if value is None:
        return "未获得"
    return f"{float(value):.4f}"


def judge_optimization_effect(original_prob, modified_prob, modified_is_toxic):
    if modified_prob is None:
        return "未获得候选分子的毒性复评结果。"

    delta = (original_prob or 0.0) - modified_prob

    if modified_is_toxic:
        return f"候选分子 MICE TdP 风险尚未降至 low，概率变化为 {delta:.4f}，需要继续优化。"

    if delta > 0:
        return f"候选分子的 MICE TdP 概率下降 {delta:.4f}，风险方向改善。"

    return f"候选分子未降低 MICE TdP 风险，概率变化为 {delta:.4f}。"


def _build_candidate_record(
    state: AIDDState,
    target_smiles: str,
    sa_score: float,
    synthesis_feasible: bool,
    molecular_review: dict,
    constraints: dict,
):
    return {
        "round": int(state.get("optimization_round", 0) or 0),
        "smiles": target_smiles,
        "toxicity_prob": state.get("modified_toxicity_prob"),
        "is_toxic": state.get("modified_is_toxic"),
        "risk_level": state.get("modified_risk_level"),
        "channel_predictions": state.get("modified_channel_predictions", {}),
        "tdp": state.get("modified_tdp", {}),
        "sa_score": sa_score,
        "synthesis_feasible": synthesis_feasible,
        "molecular_review": molecular_review,
        "constraints": constraints,
    }


async def agent_synthesis_evaluator(state: AIDDState):
    print(">>> Node 3 [Synthesis evaluator] started")

    target_smiles = state.get("modified_smiles") or state["original_smiles"]
    current_round = int(state.get("optimization_round", 0) or 0)
    max_rounds = int(state.get("max_optimization_rounds", 3) or 3)

    sa_score = calculate_sa_score(target_smiles)
    evaluation = describe_sa_score(sa_score)
    synthesis_feasible = sa_score < SYNTHESIS_SA_THRESHOLD
    molecular_review = review_candidate(state["original_smiles"], target_smiles)

    original_toxicity_prob = state.get("toxicity_prob")
    original_risk_level = state.get("risk_level", "unknown")
    toxicity_source = state.get("toxicity_source", "unknown")

    if current_round == 0:
        candidate_prob = original_toxicity_prob
        candidate_is_toxic = state.get("is_toxic")
        candidate_risk_level = original_risk_level
        candidate_toxicity_source = toxicity_source
    else:
        candidate_prob = state.get("modified_toxicity_prob")
        candidate_is_toxic = state.get("modified_is_toxic")
        candidate_risk_level = state.get("modified_risk_level")
        candidate_toxicity_source = state.get("modified_toxicity_source", "unknown")

    toxicity_ok = candidate_is_toxic is False
    original_channels = state.get("channel_predictions", {})
    candidate_channels = (
        state.get("modified_channel_predictions", {})
        if current_round > 0
        else original_channels
    )
    original_herg_ic50 = original_channels.get("herg", {}).get("ic50_uM")
    candidate_herg_ic50 = candidate_channels.get("herg", {}).get("ic50_uM")
    original_nav_ic50 = original_channels.get("nav1.5", {}).get("ic50_uM")
    candidate_nav_ic50 = candidate_channels.get("nav1.5", {}).get("ic50_uM")

    herg_improved = current_round == 0 or (
        original_herg_ic50 is not None
        and candidate_herg_ic50 is not None
        and float(candidate_herg_ic50) > float(original_herg_ic50)
    )
    nav15_preserved = current_round == 0 or (
        original_nav_ic50 is not None
        and candidate_nav_ic50 is not None
        and float(candidate_nav_ic50) >= float(original_nav_ic50)
    )
    drug_like = molecular_review["drug_likeness"]["passed"]
    similarity_preserved = molecular_review["structure_similarity"]["passed"]
    activity_proxy_preserved = molecular_review["activity_retention_proxy"]["passed"]
    constraints = {
        "tdp_low_risk": toxicity_ok,
        "herg_ic50_improved": herg_improved,
        "nav1.5_ic50_preserved": nav15_preserved,
        "drug_likeness_passed": drug_like,
        "structure_similarity_passed": similarity_preserved,
        "activity_proxy_passed": activity_proxy_preserved,
        "synthesis_feasible": synthesis_feasible,
    }
    optimization_success = all(constraints.values())
    reached_limit = current_round >= max_rounds
    optimization_failed = bool(not optimization_success and reached_limit)

    if optimization_success:
        failure_reason = None
        decision = "成功：候选分子通过三通道、类药性、相似性和合成可行性复审。"
    elif optimization_failed:
        failure_reason = (
            f"已完成 {current_round}/{max_rounds} 轮优化，但候选仍未通过全部多目标约束。"
        )
        decision = "失败：达到最大优化轮数，未获得通过全部复审条件的候选分子。"
    else:
        unmet = []
        if not toxicity_ok:
            unmet.append("MICE TdP 风险尚未降至 low")
        if not herg_improved:
            unmet.append("hERG IC50 未提升")
        if not synthesis_feasible:
            unmet.append(f"SA Score={sa_score}，合成难度偏高")
        if not nav15_preserved:
            unmet.append("Nav1.5 IC50 降低，存在传导减慢/QRS 延长约束风险")
        if not drug_like:
            unmet.append("QED/Lipinski 类药性未通过")
        if not similarity_preserved:
            unmet.append("与原分子的结构相似度低于 0.40")
        failure_reason = "；".join(unmet) or "候选尚未满足成功条件"
        decision = f"继续优化：{failure_reason}。"

    history = state.get("optimization_history", [])
    optimization_text = "\n".join(f"- {item}" for item in history) if history else "未触发结构优化。"

    candidate_history = list(state.get("candidate_history", []))
    if current_round > 0:
        candidate_history.append(
            _build_candidate_record(
                state,
                target_smiles,
                sa_score,
                synthesis_feasible,
                molecular_review,
                constraints,
            )
        )

    trace = list(state.get("agent_trace", []))
    trace.append(
        {
            "round": current_round,
            "agent": "合成评估与决策智能体",
            "status": "completed" if optimization_success else ("failed" if optimization_failed else "retry"),
            "message": decision,
            "smiles": target_smiles,
        }
    )

    optimization_effect = judge_optimization_effect(
        original_prob=original_toxicity_prob,
        modified_prob=state.get("modified_toxicity_prob"),
        modified_is_toxic=state.get("modified_is_toxic"),
    )

    final_report = (
        "【最终评估报告】\n"
        f"任务结论：{decision}\n"
        f"优化轮次：{current_round}/{max_rounds}\n\n"
        "【分子信息】\n"
        f"原始分子：{state['original_smiles']}\n"
        f"当前候选分子：{target_smiles}\n\n"
        "【毒性评估】\n"
        f"原始 MICE TdP 概率：{_format_prob(original_toxicity_prob)}\n"
        f"原始风险等级：{original_risk_level}\n"
        f"当前候选 MICE TdP 概率：{_format_prob(candidate_prob)}\n"
        f"当前候选风险等级：{candidate_risk_level}\n"
        f"毒性模型来源：{candidate_toxicity_source}\n"
        f"毒性优化判断：{optimization_effect}\n\n"
        f"hERG IC50 提升约束：{'满足' if herg_improved else '不满足'}\n"
        f"Nav1.5 IC50 保持约束：{'满足' if nav15_preserved else '不满足'}\n\n"
        "【活性保持代理与类药性】\n"
        f"Morgan-Tanimoto 相似度：{molecular_review['structure_similarity']['score']:.4f}\n"
        f"活性保持代理：{'通过' if activity_proxy_preserved else '不通过'}（仅为结构相似性代理，非真实活性预测）\n"
        f"QED：{molecular_review['drug_likeness']['qed']:.4f}\n"
        f"Lipinski 违例数：{molecular_review['drug_likeness']['lipinski_violation_count']}\n"
        f"类药性判断：{'通过' if drug_like else '不通过'}\n\n"
        "【合成可行性】\n"
        f"SA Score：{sa_score}（{evaluation}）\n"
        f"合成可行性判断：{'可合成' if synthesis_feasible else '合成难度偏高'}\n\n"
        "【优化记录】\n"
        f"{optimization_text}\n\n"
        "【RAG 证据摘要：原始分子】\n"
        f"{summarize_herg_context(state.get('rag_context', []))}\n\n"
        "【RAG 证据摘要：候选分子】\n"
        f"{summarize_herg_context(state.get('modified_rag_context', []))}\n"
    )

    return {
        "sa_score": sa_score,
        "synthesis_feasible": synthesis_feasible,
        "molecular_review": molecular_review,
        "optimization_success": optimization_success,
        "optimization_failed": optimization_failed,
        "failure_reason": failure_reason,
        "candidate_history": candidate_history,
        "agent_trace": trace,
        "final_report": final_report,
    }
