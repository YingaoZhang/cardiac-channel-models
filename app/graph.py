from langgraph.graph import StateGraph, END

from app.state import AIDDState
from app.agents.toxicity_agent import agent_toxicity_assessor
from app.agents.chemist_agent import agent_molecule_chemist
from app.agents.retoxicity_agent import agent_modified_toxicity_assessor
from app.agents.synthesis_agent import agent_synthesis_evaluator


MAX_OPTIMIZATION_ROUNDS = 3


def route_after_toxicity(state: AIDDState):
    if state.get("is_toxic"):
        return "Optimization"

    return "Synthesis"


def route_after_synthesis(state: AIDDState):
    if state.get("optimization_success"):
        return "End"

    current_round = int(state.get("optimization_round", 0) or 0)
    max_rounds = int(state.get("max_optimization_rounds", MAX_OPTIMIZATION_ROUNDS) or MAX_OPTIMIZATION_ROUNDS)
    if current_round < max_rounds:
        return "Optimization"

    return "End"


def build_graph():
    workflow = StateGraph(AIDDState)

    workflow.add_node("Toxicity", agent_toxicity_assessor)
    workflow.add_node("Optimization", agent_molecule_chemist)
    workflow.add_node("ReToxicity", agent_modified_toxicity_assessor)
    workflow.add_node("Synthesis", agent_synthesis_evaluator)

    workflow.set_entry_point("Toxicity")

    workflow.add_conditional_edges(
        "Toxicity",
        route_after_toxicity,
        {
            "Optimization": "Optimization",
            "Synthesis": "Synthesis",
        },
    )

    workflow.add_edge("Optimization", "ReToxicity")
    workflow.add_edge("ReToxicity", "Synthesis")
    workflow.add_conditional_edges(
        "Synthesis",
        route_after_synthesis,
        {
            "Optimization": "Optimization",
            "End": END,
        },
    )

    return workflow.compile()


aidd_app = build_graph()
