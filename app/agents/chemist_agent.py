from app.services.llm_chemist import optimize_molecule_by_llm
from app.state import AIDDState


async def agent_molecule_chemist(state: AIDDState):
    current_round = int(state.get("optimization_round", 0) or 0) + 1
    max_rounds = int(state.get("max_optimization_rounds", 3) or 3)
    source_smiles = state.get("modified_smiles") or state["original_smiles"]
    toxicity_prob = (
        state.get("modified_toxicity_prob")
        if state.get("modified_toxicity_prob") is not None
        else state.get("toxicity_prob")
    ) or 0.0
    rag_context = state.get("modified_rag_context") or state.get("rag_context", [])

    print(
        f">>> Node 2 [Medicinal chemist] round {current_round}/{max_rounds} started. "
        f"hERG probability: {toxicity_prob:.4f}"
    )

    max_retry = 2
    last_error = None

    for attempt in range(1, max_retry + 1):
        try:
            print(f"    Requesting LLM structure optimization, attempt {attempt}...")

            llm_result = await optimize_molecule_by_llm(
                smiles=source_smiles,
                toxicity_prob=toxicity_prob,
                rag_context=rag_context,
                original_channel_predictions=state.get("channel_predictions", {}),
                current_channel_predictions=(
                    state.get("modified_channel_predictions")
                    or state.get("channel_predictions", {})
                ),
                optimization_round=current_round,
                max_optimization_rounds=max_rounds,
                previous_failure=state.get("failure_reason"),
            )

            new_smiles = llm_result.get("new_smiles") or source_smiles
            rationale = llm_result.get("rationale") or "模型未提供明确优化理由。"

            print(f"    Structure optimization finished. New SMILES: {new_smiles}")

            history = list(state.get("optimization_history", []))
            history.append(f"第 {current_round} 轮：{rationale}")
            trace = list(state.get("agent_trace", []))
            trace.append(
                {
                    "round": current_round,
                    "agent": "药物化学优化智能体",
                    "status": "completed",
                    "message": f"提出第 {current_round} 轮候选分子。",
                    "smiles": new_smiles,
                }
            )

            return {
                "optimization_round": current_round,
                "max_optimization_rounds": max_rounds,
                "modified_smiles": new_smiles,
                "optimization_history": history,
                "agent_trace": trace,
            }

        except Exception as exc:
            last_error = exc
            print(f"[Medicinal chemist] optimization attempt {attempt} failed: {exc}")

    new_smiles = source_smiles
    rationale = f"第 {current_round} 轮结构优化失败，保留上一轮分子。错误：{last_error}"

    history = list(state.get("optimization_history", []))
    history.append(rationale)
    trace = list(state.get("agent_trace", []))
    trace.append(
        {
            "round": current_round,
            "agent": "药物化学优化智能体",
            "status": "failed",
            "message": rationale,
            "smiles": new_smiles,
        }
    )

    return {
        "optimization_round": current_round,
        "max_optimization_rounds": max_rounds,
        "modified_smiles": new_smiles,
        "optimization_history": history,
        "agent_trace": trace,
    }
