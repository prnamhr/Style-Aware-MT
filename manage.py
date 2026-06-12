#!/usr/bin/env python
"""Command dispatcher for the Style-Aware-MT data pipeline.

Django-style entry point: `python manage.py <command> [args...]`.
Each command delegates to the `main()` of the underlying module, which keeps
its own argparse — so all the existing `--flags` work verbatim, e.g.:

    python manage.py preprocess
    python manage.py preprocess --output_dir data/processed
    python manage.py split --seed 7
"""

import importlib
import sys

# command name -> module exposing a main() entry point
COMMANDS = {
    "preprocess": "src.data.preprocess",
    "split": "src.data.split",
    "build_index": "src.retrieval.build_index",
    "infer": "src.infer.run",
    "eval": "src.eval.quick",
    "fertility": "src.eval.fertility",
    "stylometrics": "src.eval.stylometrics",
}


def _usage() -> str:
    lines = ["usage: python manage.py <command> [args...]", "", "available commands:"]
    width = max(len(name) for name in COMMANDS)
    for name, module in sorted(COMMANDS.items()):
        lines.append(f"  {name:<{width}}  -> {module}")
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(_usage())
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"error: unknown command '{command}'\n", file=sys.stderr)
        print(_usage(), file=sys.stderr)
        sys.exit(2)

    # Drop the subcommand so the target module's argparse sees only its own args.
    sys.argv = [f"manage.py {command}", *sys.argv[2:]]
    module = importlib.import_module(COMMANDS[command])
    module.main()


if __name__ == "__main__":
    main()
