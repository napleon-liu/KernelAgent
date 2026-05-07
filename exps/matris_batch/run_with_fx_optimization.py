#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path('/share/home/liuyao/workspace/KernelAgent')
INPUT_ROOT = ROOT / 'exps/matris_batch/optimization_inputs'
RESULT_ROOT = ROOT / 'exps/matris_batch/optimization_results'
SUMMARY_CSV = RESULT_ROOT / 'summary.csv'
SUMMARY_JSON = RESULT_ROOT / 'summary.json'
PREPARED = INPUT_ROOT / 'prepared_with_fx_success.json'


def parse_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text)
    if not match:
        return None
    return float(match.group(1))


def classify_error(stdout: str, stderr: str) -> str | None:
    text = stdout + '\n' + stderr
    if '504 Gateway Timeout' in text:
        return 'llm_504_gateway_timeout'
    if 'Initial kernel failed correctness verification' in text:
        return 'initial_correctness_failed'
    if 'NCU binary not found' in text:
        return 'ncu_not_found'
    if 'no successful workers' in text.lower():
        return 'no_successful_workers'
    if 'timed out' in text.lower():
        return 'timeout'
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--max-rounds', type=int, default=5)
    parser.add_argument('--strategy', default='greedy')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    data = json.loads(PREPARED.read_text())
    ops = data['prepared']
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    done = 0
    for op in ops:
        status_path = RESULT_ROOT / op / 'status.json'
        if status_path.exists() and not args.force:
            try:
                st = json.loads(status_path.read_text())
                if st.get('completed') and st.get('success'):
                    continue
            except Exception:
                pass
        out_dir = RESULT_ROOT / op
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / 'examples/run_opt_manager.py'),
            '--kernel-dir', str(INPUT_ROOT / op),
            '--strategy', args.strategy,
            '--max-rounds', str(args.max_rounds),
        ]
        print(f'[opt] {op}: {" ".join(cmd)}', flush=True)
        status = {'operator_id': op, 'completed': False, 'cmd': cmd}
        status_path.write_text(json.dumps(status, indent=2))
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=3600)
        stdout = proc.stdout
        stderr = proc.stderr
        success = (
            proc.returncode == 0
            and f'{args.strategy.upper()} OPTIMIZATION SUCCESSFUL!' in stdout
            and f'{args.strategy.upper()} OPTIMIZATION FAILED' not in stdout
        )
        best_time_ms = parse_float(r'Best time: ([0-9.]+)\s*ms', stdout + '\n' + stderr)
        initial_time_ms = parse_float(r'Initial kernel time: ([0-9.]+)ms', stderr)
        pytorch_time_ms = parse_float(r'PyTorch baseline: ([0-9.]+)ms', stderr)
        compile_time_ms = parse_float(r'PyTorch compile baseline: ([0-9.]+)ms', stderr)
        speedup_initial = parse_float(r'Speedup vs initial kernel: ([0-9.]+)x', stderr)
        speedup_pytorch = parse_float(r'Speedup vs PyTorch eager: ([0-9.]+)x', stderr)
        total_rounds = parse_float(r'Total rounds: ([0-9]+)', stdout)
        optimized_kernel = INPUT_ROOT / op / f'optimized_kernel_{args.strategy.lower()}.py'
        status.update({
            'completed': True,
            'returncode': proc.returncode,
            'success': success,
            'error_kind': None if success else classify_error(stdout, stderr),
            'best_time_ms': best_time_ms,
            'initial_time_ms': initial_time_ms,
            'pytorch_time_ms': pytorch_time_ms,
            'compile_time_ms': compile_time_ms,
            'speedup_vs_initial': speedup_initial,
            'speedup_vs_pytorch': speedup_pytorch,
            'total_rounds': int(total_rounds) if total_rounds is not None else None,
            'optimized_kernel': str(optimized_kernel) if optimized_kernel.exists() else None,
            'stdout_tail': stdout[-8000:],
            'stderr_tail': stderr[-8000:],
        })
        status_path.write_text(json.dumps(status, indent=2))
        (out_dir / 'stdout.log').write_text(proc.stdout)
        (out_dir / 'stderr.log').write_text(proc.stderr)
        done += 1
        if args.limit is not None and done >= args.limit:
            break
    rows = []
    for op in ops:
        status_path = RESULT_ROOT / op / 'status.json'
        if not status_path.exists():
            rows.append({'operator_id': op, 'completed': False, 'success': False})
            continue
        st = json.loads(status_path.read_text())
        rows.append({
            'operator_id': op,
            'completed': st.get('completed'),
            'success': st.get('success'),
            'error_kind': st.get('error_kind'),
            'best_time_ms': st.get('best_time_ms'),
            'initial_time_ms': st.get('initial_time_ms'),
            'pytorch_time_ms': st.get('pytorch_time_ms'),
            'compile_time_ms': st.get('compile_time_ms'),
            'speedup_vs_initial': st.get('speedup_vs_initial'),
            'speedup_vs_pytorch': st.get('speedup_vs_pytorch'),
            'total_rounds': st.get('total_rounds'),
            'optimized_kernel': st.get('optimized_kernel'),
        })
    summary = {
        'processed_this_run': done,
        'total': len(rows),
        'completed': sum(1 for r in rows if r.get('completed')),
        'success': sum(1 for r in rows if r.get('success')),
        'failed': sum(1 for r in rows if r.get('completed') and not r.get('success')),
        'pending': sum(1 for r in rows if not r.get('completed')),
        'result_root': str(RESULT_ROOT),
        'rows': rows,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    with SUMMARY_CSV.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
