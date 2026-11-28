"""Generic runnable agent executor (Req 8) supporting any AgentDefinition."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from forgeai.config import ForgeSettings, get_settings
from forgeai.ingestion.chunker import DocumentChunk, chunk_corpus
from forgeai.ingestion.loader import LoadedCorpus, load_corpus
from forgeai.llm.base import LLMProvider, parse_json_dict
from forgeai.llm.router import build_provider
from forgeai.retrieval.hybrid import HybridRetriever, RetrievedChunk
from forgeai.schemas.agent import AgentDefinition, AgentResponse
from forgeai.tools.registry import ToolAuthorizationError, execute_tool, format_tools_for_prompt


@dataclass
class ExecutionTrace:
    """Full execution telemetry returned alongside the validated AgentResponse."""

    response: AgentResponse
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)
    raw_contract_valid: bool = True
    contract_repaired: bool = False
    latency_ms: float = 0.0


class AgentRuntime:
    """Generic executor that loads ANY ForgeAI AgentDefinition and runs queries with RAG + tools."""

    def __init__(
        self,
        agent: AgentDefinition,
        llm: LLMProvider,
        *,
        corpus: LoadedCorpus | None = None,
        docs_dir_override: Path | None = None,
        embed_model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.agent = agent
        self.llm = llm
        self.embed_model_name = embed_model_name
        self.docs_dir = Path(docs_dir_override or agent.docs_dir).resolve()
        self.corpus = corpus or load_corpus(self.docs_dir)
        self._retriever: HybridRetriever | None = None
        self._indexed_signature: tuple[int, int, str] | None = None
        self._build_index()

    @classmethod
    def from_directory(
        cls,
        agent_dir: str | Path,
        *,
        provider: str | None = None,
        model_name: str | None = None,
        gemini_api_key: str | None = None,
        ollama_host: str | None = None,
        docs_dir_override: Path | None = None,
        settings: ForgeSettings | None = None,
    ) -> "AgentRuntime":
        """Load an agent bundle directory containing agent_definition.yaml or .json."""
        cfg = settings or get_settings()
        root = Path(agent_dir).resolve()
        yaml_path = root / "agent_definition.yaml"
        json_path = root / "agent_definition.json"

        if yaml_path.exists():
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        elif json_path.exists():
            data = json.loads(json_path.read_text(encoding="utf-8"))
        elif root.is_file() and root.suffix.lower() in {".yaml", ".yml", ".json"}:
            raw = root.read_text(encoding="utf-8")
            data = yaml.safe_load(raw) if root.suffix.lower() != ".json" else json.loads(raw)
        else:
            raise FileNotFoundError(
                f"Could not find agent_definition.yaml or agent_definition.json in {root}"
            )

        agent = AgentDefinition.model_validate(data)
        resolved_docs = docs_dir_override
        if resolved_docs is None:
            cand_raw = Path(agent.docs_dir)
            if not cand_raw.exists():
                sibling_docs = (root.parent if root.is_dir() else root.parent.parent) / "docs"
                if sibling_docs.exists():
                    resolved_docs = sibling_docs
        chosen_provider = provider or agent.provider
        if chosen_provider.startswith("gemini"):
            chosen_provider = "gemini"
        chosen_model = model_name or agent.model_name
        llm = build_provider(
            provider=chosen_provider,
            model_name=chosen_model,
            gemini_api_key=gemini_api_key,
            ollama_host=ollama_host,
            settings=cfg,
        )
        return cls(
            agent,
            llm,
            docs_dir_override=resolved_docs,
            embed_model_name=cfg.embed_model,
        )

    def _build_index(self) -> None:
        sig = (
            self.agent.retrieval.chunk_size,
            self.agent.retrieval.chunk_overlap,
            self.agent.retrieval.chunking_strategy,
        )
        if self._retriever is not None and self._indexed_signature == sig:
            return
        chunks: list[DocumentChunk] = chunk_corpus(
            self.corpus,
            chunk_size=self.agent.retrieval.chunk_size,
            chunk_overlap=self.agent.retrieval.chunk_overlap,
            strategy=self.agent.retrieval.chunking_strategy,
        )
        self._retriever = HybridRetriever(
            chunks,
            embed_model_name=self.embed_model_name,
            use_dense=True,
        )
        self._indexed_signature = sig

    def update_agent(self, new_agent: AgentDefinition) -> None:
        """Update agent definition in-place and rebuild index only if chunking changed."""
        self.agent = new_agent
        self._build_index()

    def _format_context(self, hits: list[RetrievedChunk]) -> str:
        if not hits:
            return (
                "[NO RETRIEVED CHUNKS PASSED RELEVANCE THRESHOLD. "
                "If tools do not provide direct evidence, you MUST refuse.]"
            )
        parts = []
        for h in hits:
            parts.append(f"(score={h.score:.3f}) {h.chunk.format_for_prompt()}")
        return "\n\n---\n\n".join(parts)

    def _parse_or_repair_response(
        self,
        raw_text: str,
        *,
        tools_used: list[str],
        hits: list[RetrievedChunk],
    ) -> tuple[AgentResponse, bool, bool]:
        """Parse raw output against AgentResponse contract; return (resp, raw_valid, repaired)."""
        try:
            data = parse_json_dict(raw_text)
            if "answer" not in data and "refusal_reason" in data:
                data["answer"] = str(data.get("refusal_reason") or self.agent.output_contract.refusal_phrase_prefix)
            if "answer" not in data:
                raise ValueError("Missing 'answer' key")

            # Normalize citations if small model returned list of strings
            cits = data.get("citations") or []
            norm_cits: list[dict[str, str]] = []
            if isinstance(cits, list):
                for item in cits:
                    if isinstance(item, dict):
                        src = str(item.get("source") or item.get("doc") or item.get("file") or "").strip()
                        exc = str(item.get("excerpt") or item.get("quote") or "").strip()
                        if src:
                            norm_cits.append({"source": src, "excerpt": exc[:180]})
                    elif isinstance(item, str) and item.strip():
                        norm_cits.append({"source": item.strip(), "excerpt": ""})
            data["citations"] = norm_cits

            # Merge actual tool calls executed by runtime
            model_tools = data.get("tools_used") if isinstance(data.get("tools_used"), list) else []
            merged_tools = list(dict.fromkeys([str(t) for t in tools_used] + [str(t) for t in model_tools]))
            data["tools_used"] = merged_tools

            if data.get("confidence") not in {"high", "medium", "low"}:
                data["confidence"] = "medium"

            resp = AgentResponse.model_validate(data)
            if not resp.refused and re.search(
                r"\b(cannot answer|unable to answer|cannot be (?:made|provided|determined|answered)|cannot approve|do(?:es)? not contain|not available in|not covered in the provided|outside (?:the |my )?scope|cannot verify)\b",
                resp.answer,
                re.I,
            ):
                resp.refused = True
                if not resp.refusal_reason:
                    resp.refusal_reason = "Refusal detected in answer text."
            return resp, True, False
        except Exception:
            # Repair fallback while honestly recording raw_contract_valid=False
            clean = raw_text.strip() or self.agent.output_contract.refusal_phrase_prefix
            looks_refused = bool(
                re.search(
                    r"\b(cannot answer|unable to answer|cannot be (?:made|provided|determined|answered)|cannot approve|do(?:es)? not contain|not available in|not covered|not mentioned|no information|out of scope)\b",
                    clean,
                    re.I,
                )
            )
            default_cits = (
                [{"source": hits[0].chunk.source, "excerpt": hits[0].chunk.text[:100]}]
                if (hits and not looks_refused)
                else []
            )
            resp = AgentResponse(
                answer=clean,
                citations=default_cits,
                refused=looks_refused,
                refusal_reason="Repaired non-JSON output" if looks_refused else None,
                confidence="low",
                tools_used=tools_used,
            )
            return resp, False, True

    def ask(
        self,
        question: str,
        *,
        chat_history: list[dict[str, str]] | None = None,
    ) -> ExecutionTrace:
        """Execute retrieval, optional multi-step tool calling, and output contract validation."""
        assert self._retriever is not None
        t0 = time.perf_counter()

        hits = self._retriever.search_with_decomposition(
            question, self.agent.retrieval, apply_threshold=True
        )
        context_block = self._format_context(hits)
        tools_block = format_tools_for_prompt(self.agent.tools, custom_tools=self.agent.custom_tools)
        has_any_tools = bool(self.agent.tools or self.agent.custom_tools)

        history_str = ""
        if chat_history:
            recent = chat_history[-4:]
            history_str = "\nCONVERSATION HISTORY:\n" + "\n".join(
                f"{m.get('role', 'user').upper()}: {m.get('content', '')[:300]}"
                for m in recent
            )

        contract_instruction = (
            "RESPONSE PROTOCOL:\n"
            "Return ONLY valid JSON.\n"
            "Option A (ONLY if an authorized tool is needed before answering):\n"
            '{"action": "tool_call", "tool": "<authorized_tool_name>", "args": {...}}\n'
            "Option B (Final answer matching output contract):\n"
            '{"action": "final", "answer": "<concise grounded answer or refusal>", '
            '"citations": [{"source": "<exact filename>", "excerpt": "<short supporting quote>"}], '
            '"refused": false, "refusal_reason": null, "confidence": "high|medium|low", "tools_used": []}'
        )

        system_msg = f"{self.agent.system_prompt}\n\nAUTHORIZED TOOLS:\n{tools_block}\n\n{contract_instruction}"
        user_msg = f"{history_str}\nRETRIEVED CONTEXT:\n{context_block}\n\nUSER QUESTION:\n{question}"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        tool_outputs: list[dict[str, Any]] = []
        tools_used: list[str] = []
        final_raw = ""

        max_steps = self.agent.max_tool_steps if has_any_tools else 1
        for step in range(max_steps + 1):
            llm_resp = self.llm.complete(
                messages,
                temperature=self.agent.temperature,
                max_tokens=900,
                json_mode=True,
            )
            final_raw = llm_resp.text

            try:
                parsed = parse_json_dict(final_raw)
            except Exception:
                # Try one strict JSON format repair prompt
                messages.append({"role": "assistant", "content": final_raw[:800]})
                messages.append(
                    {
                        "role": "user",
                        "content": 'Format your answer STRICTLY as JSON: {"action":"final","answer":"...","citations":[{"source":"filename","excerpt":"..."}],"refused":false,"refusal_reason":null,"confidence":"medium","tools_used":[]}',
                    }
                )
                repair_resp = self.llm.complete(
                    messages,
                    temperature=0.0,
                    max_tokens=800,
                    json_mode=True,
                )
                final_raw = repair_resp.text
                break

            action = str(parsed.get("action") or "").lower().strip()
            if action == "tool_call" and step < max_steps and has_any_tools:
                t_name = str(parsed.get("tool") or "").strip()
                t_args = parsed.get("args") if isinstance(parsed.get("args"), dict) else {}
                try:
                    obs = execute_tool(
                        t_name,
                        t_args,
                        authorized_tools=self.agent.tools,
                        docs_dir=self.docs_dir,
                        custom_tools=self.agent.custom_tools,
                    )
                    tools_used.append(t_name)
                except ToolAuthorizationError as exc:
                    obs = json.dumps({"error": str(exc)})
                except Exception as exc:
                    obs = json.dumps({"error": f"Tool execution error: {exc}"})

                tool_outputs.append({"tool": t_name, "args": t_args, "output": obs})
                messages.append({"role": "assistant", "content": json.dumps(parsed)})
                follow_up = (
                    "Use ONLY exact values from TOOL OBSERVATION and RETRIEVED CONTEXT. If another authorized tool (e.g. calculator) is needed, return Option A; otherwise return Option B final JSON with exact grounded answer and citations."
                    if step < max_steps - 1
                    else "Use ONLY exact values from TOOL OBSERVATION and RETRIEVED CONTEXT. Now return Option B final JSON with exact grounded answer and citations."
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"TOOL OBSERVATION ({t_name}):\n{obs[:1800]}\n\n{follow_up}",
                    }
                )
                continue

            break

        resp, raw_valid, repaired = self._parse_or_repair_response(
            final_raw,
            tools_used=tools_used,
            hits=hits,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return ExecutionTrace(
            response=resp,
            retrieved_chunks=hits,
            tool_outputs=tool_outputs,
            raw_contract_valid=raw_valid,
            contract_repaired=repaired,
            latency_ms=latency_ms,
        )
