import json
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from rdkit import Chem

from app.services.rag_service import summarize_herg_context
from app.settings import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
)


def response_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    raise ValueError(f"Unsupported LLM response content type: {type(content).__name__}")


def clean_json_text(content: Any) -> str:
    content = response_text(content).strip()

    if content.startswith("```json"):
        content = content[7:]

    if content.startswith("```"):
        content = content[3:]

    if content.endswith("```"):
        content = content[:-3]

    match = re.search(r"\{.*\}", content.strip(), re.DOTALL)
    if match:
        return match.group(0).strip()

    return content.strip()


def validate_smiles(smiles: str) -> bool:
    if not smiles:
        return False

    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


async def optimize_molecule_by_llm(
    smiles: str,
    toxicity_prob: float,
    rag_context: list[dict[str, Any]] | None = None,
    original_channel_predictions: dict[str, Any] | None = None,
    current_channel_predictions: dict[str, Any] | None = None,
    optimization_round: int = 1,
    max_optimization_rounds: int = 3,
    previous_failure: str | None = None,
) -> dict:
    if not LLM_API_KEY:
        raise ValueError("Missing LLM_API_KEY. Configure it in the .env file.")

    llm = ChatOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
你是一名谨慎的药物化学优化智能体。你的任务是在尽量保留主体骨架的前提下，提出一个新的候选分子，用于降低三通道心脏安全风险并保持合成可行性。

设计目标：
1. 优先提高 hERG IC50（降低 hERG pIC50 和阻滞风险）。
2. 不得降低 Nav1.5 IC50，避免增强 Nav1.5 阻滞及传导减慢/QRS 延长风险。
3. 兼顾 Cav1.2，使 hERG/Cav1.2 通道间差分对应的 MICE Model 5 风险降低。
4. 保留主体骨架，目标 Morgan-Tanimoto 相似度不低于 0.40，以结构相似性作为活性保持代理；这不等同于真实活性预测。
5. 兼顾类药性和合成可行性：QED 目标不低于 0.40、Lipinski 违例不超过 1、SA Score 目标低于 7。
6. 参考结构相似案例和文献片段，但不要把 RAG 证据当成绝对结论。
7. 尽量避免强碱性中心、过大疏水芳香片段、过高脂溶性和明显有利于 hERG 结合的结构特征。
8. 候选分子必须是合法 SMILES；如果上一轮失败，针对失败原因做不同方向的轻量调整，不要重复同一候选。

只返回一个 JSON 对象，不要 markdown，不要额外解释。JSON 字段：
new_smiles: 优化后的候选分子 SMILES。
rationale: 中文说明，描述改了哪里、为什么可能降低 hERG 风险、为什么仍可能可合成，以及参考了哪些 RAG 证据。
                """.strip(),
            ),
            (
                "user",
                (
                    "优化轮次：{round}/{max_rounds}\n"
                    "输入分子 SMILES：{smiles}\n"
                    "当前预测 hERG 毒性概率：{prob}\n"
                    "原始三通道预测：{original_channels}\n"
                    "当前三通道预测：{current_channels}\n"
                    "上一轮失败原因：{previous_failure}\n\n"
                    "RAG 检索上下文：\n{rag_context}"
                ),
            ),
        ]
    )

    chain = prompt | llm
    response = await chain.ainvoke(
        {
            "smiles": smiles,
            "prob": toxicity_prob,
            "round": optimization_round,
            "max_rounds": max_optimization_rounds,
            "previous_failure": previous_failure or "无",
            "original_channels": json.dumps(
                original_channel_predictions or {}, ensure_ascii=False
            ),
            "current_channels": json.dumps(
                current_channel_predictions or original_channel_predictions or {},
                ensure_ascii=False,
            ),
            "rag_context": summarize_herg_context(rag_context or []),
        }
    )

    content = clean_json_text(response.content)

    try:
        llm_result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM output is not valid JSON: {content}") from exc

    new_smiles = llm_result.get("new_smiles")
    rationale = llm_result.get("rationale", "模型未提供明确理由。")

    if not validate_smiles(new_smiles):
        raise ValueError(f"LLM returned an invalid SMILES string: {new_smiles}")

    return {
        "new_smiles": new_smiles,
        "rationale": rationale,
    }
