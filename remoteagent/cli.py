"""CLI inference entrypoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from remoteagent.config.defaults import DEFAULT_VLLM_URL, ENV_URL_KEYS, SERVICE_PORTS
from remoteagent.core import RemoteAgent


class RemoteAgentCLI:
    @staticmethod
    def _default_service_url(service: str) -> str:
        env_key = ENV_URL_KEYS[service]
        return os.environ.get(env_key, f"http://127.0.0.1:{SERVICE_PORTS[service]}")

    @staticmethod
    def build_arg_parser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="RemoteAgent: vLLM + EO vision HTTP tools")
        p.add_argument("--vllm_url", type=str, default=os.environ.get("VLLM_URL", DEFAULT_VLLM_URL))
        p.add_argument("--vllm_model", type=str, default=os.environ.get("VLLM_MODEL"))
        p.add_argument(
            "--system_prompt_path",
            type=str,
            default=None,
            help="System prompt file (default: bundled remoteagent/prompts/prompt.txt)",
        )
        p.add_argument(
            "--remote_sam_url",
            type=str,
            default=RemoteAgentCLI._default_service_url("remotesam"),
        )
        p.add_argument(
            "--change3d_url",
            type=str,
            default=RemoteAgentCLI._default_service_url("change3d"),
        )
        p.add_argument(
            "--sm3det_url",
            type=str,
            default=RemoteAgentCLI._default_service_url("sm3det"),
        )
        p.add_argument(
            "--crossearth_url",
            type=str,
            default=RemoteAgentCLI._default_service_url("crossearth"),
        )
        p.add_argument(
            "--skysense_det_url",
            type=str,
            default=RemoteAgentCLI._default_service_url("skysense_det"),
        )
        p.add_argument(
            "--directsam_url",
            type=str,
            default=RemoteAgentCLI._default_service_url("directsam"),
        )
        p.add_argument("--image_path", type=str, required=True)
        p.add_argument("--max_rounds", type=int, default=3)
        p.add_argument("query", type=str, nargs="*", help="User query")
        return p

    @staticmethod
    def main(argv: list[str] | None = None) -> int:
        args = RemoteAgentCLI.build_arg_parser().parse_args(argv)
        query = " ".join(args.query).strip() if args.query else "Please describe this image."
        prompt_path = Path(args.system_prompt_path) if args.system_prompt_path else None

        api_urls = {
            "remotesam": args.remote_sam_url,
            "change3d": args.change3d_url,
            "sm3det": args.sm3det_url,
            "crossearth": args.crossearth_url,
            "skysense_det": args.skysense_det_url,
            "directsam": args.directsam_url,
        }

        agent = RemoteAgent(
            vllm_url=args.vllm_url,
            model_name=args.vllm_model,
            system_prompt_path=prompt_path,
            api_urls=api_urls,
            max_rounds=args.max_rounds,
        )

        print("=" * 60)
        print("RemoteAgent")
        print("=" * 60)
        print(f"  vLLM: {args.vllm_url} | model: {agent.resolve_model_name()}")
        for k, v in api_urls.items():
            print(f"  {k}: {v or '(not set)'}")
        print("-" * 60)
        print(f"  image: {args.image_path}")
        print(f"  query: {query[:100]}{'...' if len(query) > 100 else ''}")
        print("-" * 60)

        result = agent.run(query=query, image_path=args.image_path)
        print("\n[Final response]\n", result.text, sep="")
        print("=" * 60)
        return 0


build_arg_parser = RemoteAgentCLI.build_arg_parser
main = RemoteAgentCLI.main


__all__ = ["build_arg_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
