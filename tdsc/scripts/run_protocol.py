"""Run one frozen TDSC protocol on the released offline puzzles."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdsc_eval.answer_eval import answer_prompt, run_answers
from tdsc_eval.defense_eval import defense_prompt, run_defense_static
from tdsc_eval.interactive_eval import run_interactive
from tdsc_eval.providers import Provider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", choices=("A", "B", "D"), required=True)
    parser.add_argument("--solver", choices=("G-V", "C-S", "C-I"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", help="Local OpenAI-compatible CaptchaMind endpoint")
    parser.add_argument("--model", help="Serving identifier only; use the documented frozen checkpoint")
    parser.add_argument("--secrets", type=Path, help="Optional local credentials file; environment variables are preferred")
    parser.add_argument("--execute", action="store_true", help="Explicitly enable model calls; default is a dry run")
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT in output.parents or output == ROOT:
        parser.error("Write new runs outside the archived tdsc directory, for example ../generated/runs/a-ci.")
    names = {"A": "answer_20260913_conference", "B": "answer_20260913_external", "D": "defense_v1_eval"}
    rows = json.loads((ROOT / "manifests" / (names[args.module] + ".json")).read_text())
    config_file = "answer_20260913.json" if args.module != "D" and args.solver != "C-I" else "solver_defense_v1.json"
    config = json.loads((ROOT / "configs" / config_file).read_text())
    provider_name = "gemini" if args.solver == "G-V" else "captchamind"
    if args.model:
        config[provider_name]["model"] = args.model
    print(json.dumps({"module": args.module, "solver": args.solver, "items": len(rows),
                      "max_calls": len(rows) * (30 if args.solver == "C-I" else 1),
                      "output": str(output), "execute": args.execute}, indent=2))
    if not args.execute:
        return
    prompt = defense_prompt if args.module == "D" else answer_prompt
    provider = Provider(provider_name, config, args.secrets, args.base_url, prompt)
    if args.solver == "C-I":
        run_interactive(rows, provider, output, config=config, max_total_calls=len(rows) * 30)
    elif args.module == "D":
        run_defense_static(rows, provider, output, config, max_total_calls=len(rows), max_cost_usd=10)
    else:
        run_answers(rows, provider, output, config, workers=1, max_cost_usd=20)


if __name__ == "__main__":
    main()
