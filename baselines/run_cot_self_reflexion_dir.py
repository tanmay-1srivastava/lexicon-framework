#!/usr/bin/env python3
"""
Run baselines/cot_with_self_reflexion.py on ALL .json files in a directory,
measure runtime per file, write ONE timings txt file, and print median/std.

Example:
  python3 baselines/run_cot_self_reflexion_dir.py \
    --input-dir /path/to/jsons \
    --output-dir /path/to/outputs \
    --script baselines/cot_with_self_reflexion.py \
    --timings-file /path/to/outputs/timings.txt \
    --recursive
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, help="Directory containing input JSON files")
    p.add_argument("--output-dir", required=True, help="Directory to write outputs")
    p.add_argument("--script", required=True, help="Path to cot_with_self_reflexion.py")
    p.add_argument("--pattern", default="*.json")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-completion-tokens", type=int, default=4096)
    p.add_argument("--reflection-temperature", type=float, default=None, help="Temperature for reflection pass")
    p.add_argument("--reflection-max-completion-tokens", type=int, default=None, help="Token budget for reflection pass")
    p.add_argument(
        "--timings-file",
        default=None,
        help="Path to timings txt OR a directory (if directory, timings.txt is created inside).",
    )
    p.add_argument("--write-full", action="store_true", help="Write full output object (not just resolutions)")
    p.add_argument("--no-draft", action="store_true", help="Remove draft field from output")
    return p.parse_args()


def list_inputs(input_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    files = sorted(input_dir.rglob(pattern) if recursive else input_dir.glob(pattern))
    return [f for f in files if f.is_file()]


def out_path_for(inp: Path, input_root: Path, output_root: Path) -> Path:
    # Mirror subfolders under output_root
    rel = inp.relative_to(input_root)
    out_dir = output_root / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{inp.stem}_cot_self_reflexion_result.json"


def resolve_timings_file(timings_arg: Optional[str], output_dir: Path) -> Path:
    """
    Always returns a FILE path:
    - None -> <output_dir>/timings.txt
    - existing directory -> <dir>/timings.txt
    - trailing slash -> treat as directory-like -> <path>/timings.txt
    - otherwise -> treat as file path
    """
    if timings_arg is None:
        return output_dir / "timings.txt"

    p = Path(timings_arg).expanduser().resolve()

    if p.exists() and p.is_dir():
        return p / "timings.txt"

    if str(timings_arg).endswith(("/", "\\")):
        return p / "timings.txt"

    return p


def run_one(cmd: List[str]) -> Tuple[float, int, str]:
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    combined = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    return dt, proc.returncode, combined.strip()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    script_path = Path(args.script).expanduser().resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"--input-dir is not a directory: {input_dir}")
    if not script_path.exists():
        raise SystemExit(f"--script not found: {script_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = list_inputs(input_dir, args.pattern, args.recursive)
    if not inputs:
        raise SystemExit(f"No files matched {args.pattern} in {input_dir} (recursive={args.recursive})")

    times: List[float] = []
    lines: List[str] = []
    failures: List[Path] = []

    lines.append(f"# runner_python={sys.executable}")
    lines.append(f"# target_script={script_path}")
    lines.append(f"# input_dir={input_dir}")
    lines.append(f"# output_dir={output_dir}")
    lines.append(f"# temperature={args.temperature}")
    lines.append(f"# max_completion_tokens={args.max_completion_tokens}")
    if args.reflection_temperature is not None:
        lines.append(f"# reflection_temperature={args.reflection_temperature}")
    if args.reflection_max_completion_tokens is not None:
        lines.append(f"# reflection_max_completion_tokens={args.reflection_max_completion_tokens}")
    lines.append("# columns: seconds<TAB>status<TAB>input<TAB>output")
    lines.append("")

    for i, inp in enumerate(inputs, start=1):
        out = out_path_for(inp, input_dir, output_dir)

        if args.skip_existing and out.exists():
            print(f"[{i}/{len(inputs)}] SKIP {inp.name}")
            lines.append(f"{0.0:.6f}\tSKIP\t{inp}\t{out}")
            continue

        cmd = [
            sys.executable,
            str(script_path),
            "--input",
            str(inp),
            "--output",
            str(out),
            "--temperature",
            str(args.temperature),
            "--max-completion-tokens",
            str(args.max_completion_tokens),
        ]

        # Add reflection-specific parameters if provided
        if args.reflection_temperature is not None:
            cmd.extend(["--reflection-temperature", str(args.reflection_temperature)])
        
        if args.reflection_max_completion_tokens is not None:
            cmd.extend(["--reflection-max-completion-tokens", str(args.reflection_max_completion_tokens)])

        if args.write_full:
            cmd.append("--write-full")
        
        if args.no_draft:
            cmd.append("--no-draft")

        print(f"[{i}/{len(inputs)}] RUN  {inp.name}")
        dt, rc, msg = run_one(cmd)

        if rc == 0 and out.exists():
            times.append(dt)
            print(f"      OK   {dt:.3f}s -> {out.name}")
            lines.append(f"{dt:.6f}\tOK\t{inp}\t{out}")
        else:
            print(f"      FAIL rc={rc}  {dt:.3f}s")
            if msg:
                print("      ---- output (truncated) ----")
                print("\n".join(msg.splitlines()[:40]))
                print("      ---------------------------")
            failures.append(inp)
            lines.append(f"{dt:.6f}\tFAIL(rc={rc})\t{inp}\t{out}")

            if args.fail_fast:
                break

    # Write timings file
    timings_file = resolve_timings_file(args.timings_file, output_dir)
    timings_file.parent.mkdir(parents=True, exist_ok=True)
    timings_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved timings to: {timings_file}")

    if times:
        med = statistics.median(times)
        std = statistics.stdev(times) if len(times) > 1 else 0.0
        print("\nRuntime summary (successful only):")
        print(f"  count  = {len(times)}")
        print(f"  median = {med:.4f} s")
        print(f"  stddev = {std:.4f} s")
    else:
        print("\nNo successful runs; cannot compute median/stddev.")

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures:
            print(" -", f)
        sys.exit(2)


if __name__ == "__main__":
    main()
