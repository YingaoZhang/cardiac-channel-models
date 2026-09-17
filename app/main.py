from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.agents.chemist_agent import agent_molecule_chemist
from app.agents.retoxicity_agent import agent_modified_toxicity_assessor
from app.agents.synthesis_agent import agent_synthesis_evaluator
from app.agents.toxicity_agent import agent_toxicity_assessor
from app.graph import aidd_app
from app.services.rag_service import build_vector_index, pgvector_health
from app.services.local_toxicity import evaluate_toxicity_locally
from app.services.local_toxicity import get_local_predictor
from app.settings import (
    BGE_M3_MODEL,
    KNOWLEDGE_INDEX_PATH,
    KNOWLEDGE_RAW_DIR,
    LLM_MODEL,
    RAG_SOURCE_DIR,
)
from cardiac_channels.predictor import validate_cipa_reference_drugs


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
FRONTEND_DIST_DIR = PROJECT_DIR / "frontend" / "dist"
FRONTEND_ASSETS_DIR = FRONTEND_DIST_DIR / "assets"

DISCOVERY_STEPS = {
    "queued": {"label": "排队中", "progress": 4},
    "Toxicity": {"label": "初始毒性评估", "progress": 18},
    "Optimization": {"label": "药物化学优化", "progress": 45},
    "ReToxicity": {"label": "候选毒性复评", "progress": 65},
    "Synthesis": {"label": "合成评估与决策", "progress": 82},
    "completed": {"label": "分析完成", "progress": 100},
    "failed": {"label": "分析失败", "progress": 100},
}

TASKS: dict[str, dict[str, Any]] = {}


app = FastAPI(title="AI Drug Discovery Multi-Agent Platform")


if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static",
    )

if FRONTEND_ASSETS_DIR.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_ASSETS_DIR)),
        name="frontend-assets",
    )


templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class MoleculeRequest(BaseModel):
    smiles: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_task(smiles: str) -> dict[str, Any]:
    task_id = uuid.uuid4().hex
    task = {
        "task_id": task_id,
        "status": "queued",
        "progress": DISCOVERY_STEPS["queued"]["progress"],
        "current_step": "queued",
        "message": DISCOVERY_STEPS["queued"]["label"],
        "smiles": smiles,
        "result": None,
        "error": None,
        "created_at": _now(),
        "updated_at": _now(),
    }
    TASKS[task_id] = task
    return task


def _update_task(task_id: str, **updates: Any) -> None:
    task = TASKS[task_id]
    task.update(updates)
    task["updated_at"] = _now()


async def _run_discovery_task(task_id: str, smiles: str) -> None:
    initial_state = {
        "original_smiles": smiles,
        "optimization_history": [],
        "optimization_round": 0,
        "max_optimization_rounds": 3,
        "candidate_history": [],
        "agent_trace": [],
    }
    merged_state: dict[str, Any] = dict(initial_state)

    try:
        await _run_task_step(task_id, merged_state, "Toxicity", agent_toxicity_assessor, 12)

        if not merged_state.get("is_toxic"):
            await _run_task_step(task_id, merged_state, "Synthesis", agent_synthesis_evaluator, 84)

        while not merged_state.get("optimization_success") and not merged_state.get("optimization_failed"):
            current_round = int(merged_state.get("optimization_round", 0) or 0)
            max_rounds = int(merged_state.get("max_optimization_rounds", 3) or 3)
            if current_round >= max_rounds:
                break

            await _run_task_step(task_id, merged_state, "Optimization", agent_molecule_chemist, 38)
            await _run_task_step(task_id, merged_state, "ReToxicity", agent_modified_toxicity_assessor, 64)
            await _run_task_step(task_id, merged_state, "Synthesis", agent_synthesis_evaluator, 84)

            if merged_state.get("optimization_success") or merged_state.get("optimization_failed"):
                break

        _update_task(
            task_id,
            status="completed",
            current_step="completed",
            message=DISCOVERY_STEPS["completed"]["label"],
            progress=100,
            result=merged_state,
        )
    except Exception as exc:
        _update_task(
            task_id,
            status="failed",
            current_step="failed",
            message=DISCOVERY_STEPS["failed"]["label"],
            progress=100,
            error=str(exc),
            result=merged_state or None,
        )


async def _run_task_step(task_id: str, state: dict[str, Any], step_name: str, handler, start_progress: int) -> None:
    step = DISCOVERY_STEPS[step_name]
    _update_task(
        task_id,
        status="running",
        current_step=step_name,
        message=step["label"],
        progress=start_progress,
        result=dict(state),
    )
    step_state = await handler(state)
    if isinstance(step_state, dict):
        state.update(step_state)
    _update_task(
        task_id,
        status="running",
        current_step=step_name,
        message=f"{step['label']}完成",
        progress=step["progress"],
        result=dict(state),
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    frontend_index = FRONTEND_DIST_DIR / "index.html"
    if frontend_index.exists():
        return FileResponse(frontend_index)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.get("/health")
async def health_check():
    pgvector = pgvector_health()
    return {
        "status": "ok",
        "message": "AI drug discovery platform is running.",
        "toxicity_backend": "xgboost-three-channel",
        "agent_model": LLM_MODEL,
        "knowledge_raw_dir": str(KNOWLEDGE_RAW_DIR),
        "rag_source_dir": str(RAG_SOURCE_DIR),
        "knowledge_index_path": str(KNOWLEDGE_INDEX_PATH),
        "knowledge_index_exists": KNOWLEDGE_INDEX_PATH.exists(),
        "rag_backend": "PostgreSQL + pgvector",
        "rag_embedding_model": BGE_M3_MODEL,
        "pgvector": pgvector,
        "vector_index_exists": pgvector["status"] == "ready" and pgvector["document_count"] > 0,
        "templates_dir": str(TEMPLATES_DIR),
        "templates_exists": TEMPLATES_DIR.exists(),
        "index_exists": (TEMPLATES_DIR / "index.html").exists(),
        "static_dir": str(STATIC_DIR),
        "static_exists": STATIC_DIR.exists(),
    }


@app.get("/api/health")
async def api_health_check():
    return await health_check()


@app.post("/api/predict")
async def api_predict(req: MoleculeRequest):
    result = await evaluate_toxicity_locally(req.smiles)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@app.get("/api/reference-validation")
async def api_reference_validation():
    return validate_cipa_reference_drugs(get_local_predictor())


@app.post("/discover")
async def run_discovery(req: MoleculeRequest):
    initial_state = {
        "original_smiles": req.smiles,
        "optimization_history": [],
        "optimization_round": 0,
        "max_optimization_rounds": 3,
        "candidate_history": [],
        "agent_trace": [],
    }

    final_state = await aidd_app.ainvoke(initial_state, config={"recursion_limit": 16})

    return final_state


@app.post("/api/discover")
async def api_run_discovery(req: MoleculeRequest):
    task = _new_task(req.smiles)
    asyncio.create_task(_run_discovery_task(task["task_id"], req.smiles))
    return task


@app.get("/api/discover/{task_id}")
async def api_get_discovery_task(task_id: str):
    task = TASKS.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Discovery task not found.")
    return task


@app.post("/rag/reindex")
async def rebuild_rag_index():
    payload = build_vector_index()
    return {
        "status": "ok",
        "document_count": payload["document_count"],
        "source_dir": payload["source_dir"],
        "model": payload["model"],
        "embedding_shape": payload["embedding_shape"],
        "database_url": payload["database_url"],
        "retrieval_mode": payload["mode"],
        "index_path": str(KNOWLEDGE_INDEX_PATH),
    }


@app.post("/api/rag/reindex")
async def api_rebuild_rag_index():
    return await rebuild_rag_index()
