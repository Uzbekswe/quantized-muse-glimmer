"""Run an asserted text-generation smoke test for a GGUF model."""

from __future__ import annotations

import argparse

from .benchmark import run_process
from .common import binary_path, command_string, extract_completion, require_file


def build_command(
    model,
    llama_cpp_dir="llama.cpp",
    prompt="Reply with exactly: smoke test passed.",
    context_length=512,
    max_tokens=16,
):
    return [
        str(binary_path(llama_cpp_dir, "llama-cli")),
        "-m",
        str(model),
        "-ngl",
        "999",
        "-c",
        str(context_length),
        "--jinja",
        "--single-turn",
        "--no-display-prompt",
        "-n",
        str(max_tokens),
        "-p",
        prompt,
    ]


def assert_smoke(stdout: str, stderr: str, returncode: int, expected: str, prompt: str = "") -> str:
    if returncode != 0:
        raise RuntimeError(f"Smoke test failed with exit code {returncode}: {stderr[-2000:]}")
    completion = extract_completion(stdout, prompt)
    if expected.casefold() not in completion.casefold():
        raise AssertionError(f"Expected smoke-test text was not generated: {expected!r}")
    return completion


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--llama-cpp-dir", default="llama.cpp")
    args = parser.parse_args(argv)

    model = require_file(args.model, "smoke-test model")
    prompt = args.prompt or f"Reply with exactly: {args.expected}"
    command = build_command(model, args.llama_cpp_dir, prompt, args.context_length, args.max_tokens)
    print(f"$ {command_string(command)}")
    stdout, stderr, returncode, _, _ = run_process(command, True)
    completion = assert_smoke(stdout, stderr, returncode, args.expected, prompt)
    print(f"Smoke test passed: {completion.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
