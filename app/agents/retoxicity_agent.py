from app.agents.toxicity_agent import _evaluate_with_fallback
from app.services.rag_service import retrieve_rag_context
from app.state import AIDDState


async def agent_modified_toxicity_assessor(state: AIDDState):
    print(">>> Node 2.5 [Modified molecule toxicity reassessor] started")

    target_smiles = state.get("modified_smiles") or state["original_smiles"]
    current_round = int(state.get("optimization_round", 0) or 0)
    rag_context = retrieve_rag_context(target_smiles)
    result = await _evaluate_with_fallback(target_smiles)
    trace = list(state.get("agent_trace", []))

    if "error" in result:
        print(f"[Modified toxicity reassessor] error: {result['error']}")
        trace.append(
            {
                "round": current_round,
                "agent": "毒性复评智能体",
                "status": "failed",
                "message": f"候选分子毒性复评失败：{result['error']}",
                "smiles": target_smiles,
            }
        )
        return {
            "modified_toxicity_prob": None,
            "modified_is_toxic": None,
            "modified_risk_level": "unknown",
            "modified_toxicity_source": result.get("source", "failed"),
            "modified_rag_context": rag_context,
            "agent_trace": trace,
        }

    if "probability" not in result:
        print(f"[Modified toxicity reassessor] invalid result format: {result}")
        trace.append(
            {
                "round": current_round,
                "agent": "毒性复评智能体",
                "status": "failed",
                "message": "候选分子毒性模型返回格式异常。",
                "smiles": target_smiles,
            }
        )
        return {
            "modified_toxicity_prob": None,
            "modified_is_toxic": None,
            "modified_risk_level": "unknown",
            "modified_toxicity_source": result.get("source", "unknown"),
            "modified_rag_context": rag_context,
            "agent_trace": trace,
        }
    trace.append(
        {
            "round": current_round,
            "agent": "毒性复评智能体",
            "status": "completed",
            "message": f"MICE Model 5 TdP 概率 {result['probability']}，风险等级 {result.get('risk_level', 'unknown')}。",
            "smiles": target_smiles,
        }
    )

    return {
        "modified_toxicity_prob": result["probability"],
        "modified_is_toxic": result["is_toxic"],
        "modified_risk_level": result.get("risk_level", "unknown"),
        "modified_channel_predictions": result.get("channels", {}),
        "modified_tdp": result.get("tdp", {}),
        "modified_toxicity_source": result.get("source", "unknown"),
        "modified_rag_context": rag_context,
        "agent_trace": trace,
    }
