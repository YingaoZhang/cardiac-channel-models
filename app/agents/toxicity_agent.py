from app.services.local_toxicity import evaluate_toxicity_locally
from app.services.rag_service import retrieve_rag_context
from app.state import AIDDState


async def _evaluate_with_fallback(smiles: str) -> dict:
    local_result = await evaluate_toxicity_locally(smiles)
    local_result["source"] = "local"
    return local_result


async def agent_toxicity_assessor(state: AIDDState):
    print(">>> Node 1 [Toxicity assessor] started")

    smiles = state["original_smiles"]
    rag_context = retrieve_rag_context(smiles)
    result = await _evaluate_with_fallback(smiles)
    trace = list(state.get("agent_trace", []))

    if "error" in result:
        print(f"[Toxicity assessor] error: {result['error']}")
        trace.append(
            {
                "round": 0,
                "agent": "初始毒性评估智能体",
                "status": "failed",
                "message": f"原始分子毒性预测失败：{result['error']}",
                "smiles": smiles,
            }
        )
        return {
            "toxicity_prob": 0.0,
            "is_toxic": False,
            "risk_level": "unknown",
            "toxicity_source": result.get("source", "failed"),
            "rag_context": rag_context,
            "agent_trace": trace,
            "final_report": f"hERG toxicity prediction failed: {result['error']}",
        }

    if "probability" not in result:
        print(f"[Toxicity assessor] invalid result format: {result}")
        trace.append(
            {
                "round": 0,
                "agent": "初始毒性评估智能体",
                "status": "failed",
                "message": "原始分子毒性模型返回格式异常。",
                "smiles": smiles,
            }
        )
        return {
            "toxicity_prob": 0.0,
            "is_toxic": False,
            "risk_level": "unknown",
            "toxicity_source": result.get("source", "unknown"),
            "rag_context": rag_context,
            "agent_trace": trace,
            "final_report": "hERG toxicity prediction returned an invalid format.",
        }
    trace.append(
        {
            "round": 0,
            "agent": "初始毒性评估智能体",
            "status": "completed",
            "message": f"原始分子 MICE Model 5 TdP 概率 {result['probability']}，风险等级 {result.get('risk_level', 'unknown')}。",
            "smiles": smiles,
        }
    )

    return {
        "toxicity_prob": result["probability"],
        "is_toxic": result["is_toxic"],
        "risk_level": result.get("risk_level", "unknown"),
        "channel_predictions": result.get("channels", {}),
        "tdp": result.get("tdp", {}),
        "toxicity_source": result.get("source", "unknown"),
        "rag_context": rag_context,
        "agent_trace": trace,
    }
