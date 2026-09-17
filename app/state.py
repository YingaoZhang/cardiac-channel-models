from typing import Any, List, Optional, TypedDict


class AIDDState(TypedDict, total=False):
    original_smiles: str

    toxicity_prob: Optional[float]
    is_toxic: Optional[bool]
    risk_level: Optional[str]
    channel_predictions: dict[str, Any]
    tdp: dict[str, Any]
    toxicity_source: Optional[str]
    rag_context: List[dict[str, Any]]

    modified_smiles: Optional[str]
    optimization_history: List[str]

    modified_toxicity_prob: Optional[float]
    modified_is_toxic: Optional[bool]
    modified_risk_level: Optional[str]
    modified_channel_predictions: dict[str, Any]
    modified_tdp: dict[str, Any]
    modified_toxicity_source: Optional[str]
    modified_rag_context: List[dict[str, Any]]
    molecular_review: dict[str, Any]

    sa_score: Optional[float]
    synthesis_feasible: Optional[bool]
    optimization_round: int
    max_optimization_rounds: int
    optimization_success: bool
    optimization_failed: bool
    failure_reason: Optional[str]
    agent_trace: List[dict[str, Any]]
    candidate_history: List[dict[str, Any]]
    final_report: str
