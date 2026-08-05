#!/usr/bin/env python
"""
Command dispatcher for the Style-Aware-MT data pipeline.

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
    "peft": "src.peft.train",
    "peft_sweep": "src.peft.sweep",
    "peft_verify": "src.peft.verify",
    "infer": "src.infer.run",
    "sweep": "src.infer.sweep",
    "afsp_sweep": "src.infer.afsp_sweep",
    "afsp_verify": "src.infer.afsp_verify",
    "eval": "src.eval.quick",
    "comet": "src.eval.comet",
    "judge": "src.eval.judge",
    "judge_batch": "src.eval.judge_batch",
    "judge_ci": "src.eval.judge_ci",
    "judge_agreement": "src.eval.judge_agreement",
    "human": "src.eval.human",
    "bootstrap": "src.eval.bootstrap",
    "fertility": "src.eval.fertility",
    "stylometrics": "src.eval.stylometrics",
    "stylometrics_ci": "src.eval.stylometrics_ci",
    "metric_agreement": "src.eval.metric_agreement",
    "peft_register": "src.eval.peft_register",
    "sweep_curves": "src.eval.sweep_curves",
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
