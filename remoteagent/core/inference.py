from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from remoteagent.config.defaults import DEFAULT_VLLM_URL, ENV_URL_KEYS
from remoteagent.core.types import AgentResult
from remoteagent.llm.vllm import VLLMChatClient
from remoteagent.parsing import ToolCallParser
from remoteagent.services import ServiceExecutor
from remoteagent.utils.text import extract_answer_tag, load_system_prompt


class RemoteAgent:
    def __init__(
        self,
        vllm_url: str = DEFAULT_VLLM_URL,
        model_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        system_prompt_path: Optional[Union[str, Path]] = None,
        api_urls: Optional[Dict[str, Optional[str]]] = None,
        max_rounds: int = 3,
        max_tokens: int = 512,
    ) -> None:
        self.vllm_url = vllm_url.rstrip("/")
        self.model_name = model_name
        self.max_rounds = max_rounds
        self.max_tokens = max_tokens
        self._parser = ToolCallParser()

        if system_prompt is not None:
            self._system_prompt = system_prompt.strip()
        else:
            p = Path(system_prompt_path) if system_prompt_path else self._default_prompt_path()
            self._system_prompt = load_system_prompt(p)

        self._api_urls = api_urls or self._api_urls_from_env()
        self._llm = VLLMChatClient(self.vllm_url)
        self._executor = ServiceExecutor(self._api_urls)

    @staticmethod
    def _api_urls_from_env() -> Dict[str, Optional[str]]:
        return {name: os.environ.get(key) for name, key in ENV_URL_KEYS.items()}

    @staticmethod
    def _default_prompt_path() -> Path:
        return Path(__file__).resolve().parents[1] / "prompts" / "prompt.txt"

    @classmethod
    def from_env(cls, **kwargs: Any) -> "RemoteAgent":
        vllm_url = kwargs.pop("vllm_url", os.environ.get("VLLM_URL", DEFAULT_VLLM_URL))
        model_name = kwargs.pop("model_name", os.environ.get("VLLM_MODEL"))
        return cls(vllm_url=vllm_url, model_name=model_name, **kwargs)

    def resolve_model_name(self) -> str:
        if self.model_name:
            return self.model_name
        discovered = VLLMChatClient.list_first_model_id(self.vllm_url)
        if discovered:
            return discovered
        return "RemoteAgent-7B"

    def run(self, query: str, image_path: Union[str, Path]) -> AgentResult:
        image_path = str(image_path).strip().strip("'").strip('"')
        final_text, history = self._agent_loop(query, image_path)
        return AgentResult(text=final_text, history=history)

    def _run_llm(self, messages: List[Dict[str, Any]]) -> str:
        return self._llm.chat(messages, model=self.resolve_model_name(), max_tokens=self.max_tokens)

    def _agent_loop(self, user_query: str, image_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        user_content = f"Image path: {image_path}\n\nUser request: {user_query}"
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

        history: List[Dict[str, Any]] = []
        final_text = ""

        for round_idx in range(self.max_rounds):
            response = self._run_llm(messages)
            history.append({"round": round_idx + 1, "response": response})

            answer_content = extract_answer_tag(response)
            if answer_content is not None:
                final_text = answer_content
                break

            parsed = self._parser.parse_tool_call(response)
            if not parsed:
                final_text = response
                break

            tool_name, tool_args = parsed
            result = self._executor.execute(tool_name, tool_args)
            final_text = result

            follow_up = (
                "Please provide a concise final response to the user based on the above results "
                "(if the user's request is already fulfilled, just summarize directly without "
                "calling more tools)."
            )
            messages.append({"role": "assistant", "content": response})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"[Tool Call] {tool_name}({json.dumps(tool_args, ensure_ascii=False)})\n"
                        f"[Execution Result]\n{result}\n\n"
                        f"{follow_up}"
                    ),
                }
            )
        else:
            response = self._run_llm(messages)
            history.append({"round": self.max_rounds, "response": response})
            answer_content = extract_answer_tag(response)
            final_text = answer_content if answer_content is not None else response

        return final_text, history
