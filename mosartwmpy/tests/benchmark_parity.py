#!/usr/bin/env python
"""Performance-parity gate benchmark for the reservoir supply kernel (issue #146).

Compares a *baseline* git ref (default: the merge base of HEAD and origin/main)
against a *candidate* ref (default: HEAD) on the bundled sample input, the same
number of simulated days, and the same thread counts, then reports:

  * mean / median wall time per model timestep for each variant, with the spread
    within a run and the spread across runs reported separately
  * the candidate/baseline ratio per thread count, with PASS/FAIL at a tolerance
    (default: the candidate may be at most 5 % slower) and the range of the
    paired per-repeat ratios, so a verdict can be read against measurement noise
  * thread scaling (1 thread -> N threads) for each variant
  * an output comparison on the state arrays in TRACKED_ARRAYS, including
    repeat-to-repeat differences within each variant (a data race shows up as a
    variant that differs from itself between identical runs)

The two refs may declare incompatible dependency pins (for example numpy<2 before
the 1.0.0 release and numpy>=2 after it), so each ref is installed into its own
virtual environment from its own git worktree. The sample dataset (~700 MB) is
downloaded once, using the candidate environment.

Typical use (Windows, from the repository root):

    py -3.12 mosartwmpy\\tests\\benchmark_parity.py --python "C:\\Python312\\python.exe" --uv --interleave --days 10 --repeats 3

Typical use (Linux/macOS, from the repository root):

    python3.12 mosartwmpy/tests/benchmark_parity.py --python python3.12 --uv --interleave --days 10 --repeats 3

Useful options: --uv installs with uv (much faster than pip, and requires uv on
PATH); --interleave alternates baseline and candidate repeats so machine drift
affects both equally; --skip-install reuses existing environments; --threads 1,8
chooses the thread counts (default: 1 and the machine's core count); --work-dir
and --data-dir place the worktrees, environments, outputs and the sample data;
--keep leaves the worktrees in place for inspection; --note adds a bullet to the
report, which is where to say what the candidate changes and so which output
differences are expected.

An existing worktree is reused only when it is already at the requested ref;
otherwise the script stops and says so, rather than measuring leftover code
under the requested ref's name. To measure a worktree deliberately left at some
other state, pass its sha as --candidate-ref.

The script needs only the standard library in the interpreter that runs it.
Everything model-related runs inside the per-variant virtual environments.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import textwrap
from pathlib import Path

# state arrays collected at the end of each run and compared across runs and variants; passed to
# the driver on the command line so this list is the single place that names them
TRACKED_ARRAYS = ('channel_storage', 'reservoir_storage', 'grid_cell_supply', 'grid_cell_deficit', 'runoff_land')

# --------------------------------------------------------------------------------------
# driver executed inside each variant's venv; must stay compatible with both code versions
# --------------------------------------------------------------------------------------
DRIVER = r'''
import json, sys, time
from pathlib import Path
import numpy as np

config_path, result_path, days = sys.argv[1], sys.argv[2], int(sys.argv[3])
tracked = sys.argv[4].split(',')

from mosartwmpy import Model
import numba

model = Model()
t0 = time.perf_counter()
model.initialize(config_path)
init_seconds = time.perf_counter() - t0

timestep = int(model.config.get('simulation.timestep'))
steps = days * (86400 // timestep)

step_seconds = []
for _ in range(steps):
    t = time.perf_counter()
    model.update()
    step_seconds.append(time.perf_counter() - t)

# a missing attribute is an error: silently dropping it would report "no differences" for an
# array nobody compared, which is exactly the signal this benchmark exists to produce
missing = [name for name in tracked if not hasattr(model.state, name)]
if missing:
    raise SystemExit(
        'model.state has no attribute(s) %s; update TRACKED_ARRAYS in benchmark_parity.py'
        % ', '.join(missing)
    )

arrays, empty = {}, []
for name in tracked:
    value = np.asarray(getattr(model.state, name), dtype=np.float64)
    if value.size:
        arrays[name] = value
    else:
        empty.append(name)
np.savez(Path(result_path).with_suffix('.npz'), **arrays)

with open(result_path, 'w') as f:
    json.dump({
        'init_seconds': init_seconds,
        'step_seconds': step_seconds,
        'steps': steps,
        'timestep': timestep,
        'empty_arrays': empty,
        'numba_threads': numba.get_num_threads(),
        'numba_threading_layer': numba.threading_layer(),
        'numba_version': numba.__version__,
        'numpy_version': np.__version__,
        'python_version': sys.version.split()[0],
    }, f)

model.finalize()
'''

CONFIG_TEMPLATE = """\
# generated by benchmark_parity.py; all input paths come from config_defaults.yaml and are
# resolved relative to the sample data directory, which is the working directory of the run
simulation:
  name: {name}
  start_date: {start}
  end_date: {end}
  output_path: {output_path}
  log_to_std_out: false
  log_to_file: false
  restart_file_frequency: yearly
"""

SAMPLE_MARKER = Path('input') / 'domains' / 'mosart_conus_nldas_grid.nc'


def run(cmd, cwd=None, env=None, check=True, capture=False):
    """Run a subprocess, echoing the command."""
    printable = ' '.join(str(c) for c in cmd)
    print(f'  $ {printable}' if cwd is None else f'  [{cwd}] $ {printable}', flush=True)
    return subprocess.run(
        [str(c) for c in cmd], cwd=cwd, env=env, check=check,
        text=True, capture_output=capture,
    )


def git(repo, *args, capture=True):
    result = run(['git', *args], cwd=repo, capture=capture)
    return result.stdout.strip() if capture else None


def venv_python(venv_dir: Path) -> Path:
    if platform.system() == 'Windows':
        return venv_dir / 'Scripts' / 'python.exe'
    return venv_dir / 'bin' / 'python'


def check_interpreter(python: str) -> str:
    out = subprocess.run(
        [python, '-c', 'import sys; print("%d.%d" % sys.version_info[:2])'],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    major, minor = (int(x) for x in out.split('.'))
    if (major, minor) < (3, 10) or (major, minor) > (3, 12):
        sys.exit(
            f'Interpreter {python} is Python {out}; both code versions require 3.10-3.12 '
            '(main allows 3.9 but the PR does not; numba has no 3.13+ wheels).'
        )
    return out


def prepare_variant(repo: Path, name: str, ref: str, work: Path, python: str, use_uv: bool, skip_install: bool,
                    extra_specs=()):
    """Create a worktree at `ref` and a venv with the package installed from it.

    `extra_specs` are additional pip requirement strings installed alongside the package, used to
    hold a variant on versions its own metadata is too loose about (e.g. the baseline predates
    pandas 3 and breaks on Arrow-backed string columns, so it is pinned to pandas<3 by default).
    """
    worktree = work / f'src-{name}'
    venv = work / f'venv-{name}'

    if not worktree.exists():
        print(f'\n== {name}: worktree at {ref}')
        git(repo, 'worktree', 'add', '--detach', str(worktree), ref, capture=False)
    else:
        # a leftover worktree would otherwise be measured in place of the requested ref, with only
        # the sha in the report to give it away
        wanted, have = git(repo, 'rev-parse', ref), git(worktree, 'rev-parse', 'HEAD')
        if wanted != have:
            sys.exit(
                f'{name}: worktree {worktree} is at {have[:9]} but --{name}-ref asks for {ref} '
                f'({wanted[:9]}). Remove it with "git worktree remove --force {worktree}", or pass '
                f'--{name}-ref {have[:9]} to measure what is already there.'
            )

    sha = git(worktree, 'rev-parse', '--short', 'HEAD')
    # tracked files only; the editable install leaves egg-info and caches behind in the worktree
    dirty = bool(git(worktree, 'status', '--porcelain', '--untracked-files=no'))
    if dirty:
        print(f'== {name}: worktree has uncommitted changes; they are part of the measurement')

    py = venv_python(venv)
    if skip_install and py.exists():
        print(f'== {name}: reusing {venv}')
        return worktree, py, sha, dirty

    specs = [s for s in extra_specs if s]
    print(f'== {name}: creating venv and installing (this is the slow part)' + (f'; extra pins {specs}' if specs else ''))
    if use_uv:
        run(['uv', 'venv', '--python', python, str(venv)])
        run(['uv', 'pip', 'install', '--python', str(py), '-e', str(worktree), *specs])
    else:
        run([python, '-m', 'venv', str(venv)])
        run([py, '-m', 'pip', 'install', '--upgrade', 'pip', 'wheel'])
        run([py, '-m', 'pip', 'install', '-e', str(worktree), *specs])
    return worktree, py, sha, dirty


def ensure_sample_data(data_dir: Path, candidate_python: Path):
    if (data_dir / SAMPLE_MARKER).exists():
        print(f'\n== sample data present in {data_dir}')
        return
    print(f'\n== downloading sample_input (~700 MB) into {data_dir} using the candidate environment')
    data_dir.mkdir(parents=True, exist_ok=True)
    run([
        candidate_python, '-c',
        'import sys; from mosartwmpy.utilities.download_data import download_data; '
        'download_data("sample_input", destination=sys.argv[1])',
        str(data_dir),
    ])
    if not (data_dir / SAMPLE_MARKER).exists():
        sys.exit(
            f'Download finished but {SAMPLE_MARKER} was not found under {data_dir}. '
            'Check the unpacked layout against the paths in mosartwmpy/config_defaults.yaml.'
        )


def write_config(work: Path, name: str, start: str, days: int) -> Path:
    from datetime import date, timedelta
    y, m, d = (int(x) for x in start.split('-'))
    end = date(y, m, d) + timedelta(days=days)
    output_path = (work / 'output' / name).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    cfg = work / f'config-{name}.yaml'
    cfg.write_text(CONFIG_TEMPLATE.format(
        name=name, start=start, end=end.isoformat(), output_path=output_path.as_posix(),
    ))
    return cfg


def thread_env(threads: int) -> dict:
    env = dict(os.environ)
    for key in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[key] = str(threads)

    # an inherited NUMBA_DISABLE_JIT would time the pure-python fallback instead of the kernels
    env.pop('NUMBA_DISABLE_JIT', None)

    return env


def run_once(py: Path, data_dir: Path, cfg: Path, result: Path, days: int, threads: int):
    driver = result.parent / 'driver.py'
    driver.write_text(DRIVER)
    run(
        [py, str(driver), str(cfg), str(result), str(days), ','.join(TRACKED_ARRAYS)],
        cwd=data_dir, env=thread_env(threads),
    )
    with open(result) as f:
        return json.load(f)


def summarize(step_seconds, skip: int):
    steady = step_seconds[skip:] if len(step_seconds) > skip + 2 else step_seconds
    return {
        'n_steps': len(step_seconds),
        'total_s': sum(step_seconds),
        'mean_s': statistics.mean(steady),
        'median_s': statistics.median(steady),
        'stdev_s': statistics.pstdev(steady) if len(steady) > 1 else 0.0,
        'first_step_s': step_seconds[0],
    }


def compare_arrays(a_path: Path, b_path: Path):
    """Max absolute and relative difference per array.

    Reading the `.npz` files needs numpy in the interpreter running this script, which is the only
    thing here that does. Without it the timing results still stand and the comparison is skipped.
    """
    try:
        import numpy as np
    except ImportError:
        return {'note': 'numpy not available in the benchmark interpreter; array comparison skipped'}
    a, b = np.load(a_path), np.load(b_path)
    out = {}
    for key in sorted(set(a.files) & set(b.files)):
        x, y = a[key], b[key]
        if x.shape != y.shape:
            out[key] = {'shape_a': list(x.shape), 'shape_b': list(y.shape)}
            continue
        finite = np.isfinite(x) & np.isfinite(y)
        diff = np.abs(x[finite] - y[finite])
        scale = np.maximum(np.abs(x[finite]), np.abs(y[finite]))
        rel = np.where(scale > 0, diff / np.where(scale > 0, scale, 1.0), 0.0)
        out[key] = {
            'max_abs': float(diff.max()) if diff.size else 0.0,
            'max_rel': float(rel.max()) if rel.size else 0.0,
            'n_differing': int((diff > 0).sum()),
            'n_nonfinite_a': int((~np.isfinite(x)).sum()),
            'n_nonfinite_b': int((~np.isfinite(y)).sum()),
        }
    return out


def fmt(x):
    return f'{x:.4f}'


def main():
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--python', required=True, help='path to a Python 3.10-3.12 interpreter used to build the two venvs')
    ap.add_argument('--baseline-ref', default=None, help='git ref of the pre-modified code (default: merge-base of HEAD and origin/main)')
    ap.add_argument('--candidate-ref', default='HEAD', help='git ref of the candidate code (default: HEAD)')
    ap.add_argument('--work-dir', default=str(repo / '.benchmark'), help='scratch directory for worktrees, venvs, outputs')
    ap.add_argument('--data-dir', default=None, help='sample data directory (default: <work-dir>/data)')
    ap.add_argument('--days', type=int, default=10, help='simulated days per run (default 10 = 80 timesteps at 3 h)')
    ap.add_argument('--start', default='1981-01-01', help='simulation start date within the 1980-1985 sample')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--threads', default=None, help='comma separated thread counts (default "1,<cpu_count>")')
    ap.add_argument('--tolerance', type=float, default=0.05, help='allowed candidate slowdown as a fraction (default 0.05)')
    ap.add_argument('--skip-steps', type=int, default=2, help='leading steps excluded from steady-state stats (compile, first file open)')
    ap.add_argument('--baseline-pins', default='pandas<3,pyarrow<19,python-benedict[io],h5py',
                    help='comma separated extra requirements for the baseline venv. Default: "pandas<3" because main predates '
                         'pandas 3 and breaks on Arrow-backed string columns; "pyarrow<19" because main pins numpy<2 and newer '
                         'pyarrow wheels refuse to import under numpy 1.x; "python-benedict[io]" and "h5py" because main uses '
                         'them without declaring them (the 1.0.0 release adds the declarations). Pass "" for none.')
    ap.add_argument('--candidate-pins', default='', help='comma separated extra requirements for the candidate venv')
    ap.add_argument('--interleave', action='store_true',
                    help='alternate baseline and candidate repeats within each thread count (B C B C ...) so slow machine '
                         'drift affects both variants equally; default runs all baseline repeats, then all candidate repeats')
    ap.add_argument('--uv', action='store_true', help='use uv for venv creation and installs')
    ap.add_argument('--skip-install', action='store_true', help='reuse existing venvs/worktrees')
    ap.add_argument('--keep', action='store_true', help='do not remove worktrees afterwards')
    ap.add_argument('--note', action='append', default=[], metavar='TEXT',
                    help='add a bullet to the report\'s Notes section, repeatable. Use it to record what the '
                         'candidate changes and so which output differences are expected, which the script '
                         'cannot know from the refs alone')
    args = ap.parse_args()

    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).resolve() if args.data_dir else work / 'data'
    threads_list = [int(t) for t in (args.threads.split(',') if args.threads else ['1', str(os.cpu_count() or 1)])]
    threads_list = sorted(set(threads_list))

    print(f'Repository: {repo}')
    print(f'Interpreter for venvs: {args.python} (Python {check_interpreter(args.python)})')

    baseline_ref = args.baseline_ref or git(repo, 'merge-base', 'HEAD', 'origin/main')
    print(f'Baseline ref:  {baseline_ref}')
    print(f'Candidate ref: {args.candidate_ref}')

    pins = {
        'baseline': [s.strip() for s in args.baseline_pins.split(',') if s.strip()],
        'candidate': [s.strip() for s in args.candidate_pins.split(',') if s.strip()],
    }
    variants = {}
    for name, ref in (('baseline', baseline_ref), ('candidate', args.candidate_ref)):
        worktree, py, sha, dirty = prepare_variant(
            repo, name, ref, work, args.python, args.uv, args.skip_install, pins[name]
        )
        variants[name] = {
            'worktree': worktree, 'python': py, 'sha': sha, 'ref': ref, 'pins': pins[name], 'dirty': dirty,
        }

    ensure_sample_data(data_dir, variants['candidate']['python'])

    configs = {name: write_config(work, name, args.start, args.days) for name in variants}
    run_dirs = {}
    for name in variants:
        for threads in threads_list:
            run_dirs[name, threads] = work / 'runs' / name / f't{threads}'
            run_dirs[name, threads].mkdir(parents=True, exist_ok=True)

    # schedule: warm-up every (variant, threads) cell first, then the timed repeats either grouped by
    # variant (default) or interleaved B C B C ... within each thread count (--interleave)
    warmup_cells = [(name, threads) for name in variants for threads in threads_list]
    if args.interleave:
        timed = [(name, threads, r) for threads in threads_list for r in range(args.repeats) for name in variants]
    else:
        timed = [(name, threads, r) for name in variants for threads in threads_list for r in range(args.repeats)]

    for name, threads in warmup_cells:
        print(f'\n== {name} @ {threads} thread(s): warm-up (compiles numba kernels into the on-disk cache)')
        run_once(variants[name]['python'], data_dir, configs[name], run_dirs[name, threads] / 'warmup.json', 1, threads)

    results = {name: {threads: [] for threads in threads_list} for name in variants}
    for name, threads, r in timed:
        print(f'== {name} @ {threads} thread(s): repeat {r + 1}/{args.repeats}' + ('  [interleaved]' if args.interleave else ''))
        data = run_once(variants[name]['python'], data_dir, configs[name], run_dirs[name, threads] / f'run{r}.json', args.days, threads)
        data['summary'] = summarize(data['step_seconds'], args.skip_steps)
        results[name][threads].append(data)

    # ----------------------------------------------------------------------------------
    # report
    # ----------------------------------------------------------------------------------
    lines = []
    lines.append('# mosartwmpy performance parity: baseline vs candidate\n')
    lines.append(f'- machine: {platform.platform()}, {os.cpu_count()} logical CPUs')
    for name, v in variants.items():
        r0 = results[name][threads_list[0]][0]
        lines.append(
            f'- {name}: `{v["ref"]}` ({v["sha"]}), python {r0["python_version"]}, '
            f'numpy {r0["numpy_version"]}, numba {r0["numba_version"]}, threading layer `{r0["numba_threading_layer"]}`'
            + (f', extra pins {v["pins"]}' if v['pins'] else '')
            + (', **worktree dirty** (uncommitted changes measured)' if v['dirty'] else '')
        )
    lines.append(f'- run: {args.days} simulated days from {args.start}, {args.repeats} repeats per cell, '
                 f'steady-state excludes the first {args.skip_steps} steps, '
                 f'order {"interleaved (B C B C ...)" if args.interleave else "grouped by variant"}')
    lines.append(f'- tolerance: candidate may be at most {args.tolerance:.0%} slower than baseline\n')

    lines.append('## Wall time per timestep (seconds, steady state)\n')
    lines.append('The two spread columns separate step-to-step variation inside one run from the '
                 'variation between whole runs. The second is what the verdict ratio rests on.\n')
    lines.append('| threads | variant | mean | median | within-run stdev | across-run stdev | total run | first step |')
    lines.append('|---|---|---|---|---|---|---|---|')
    verdicts = []
    for threads in threads_list:
        run_means = {}
        for name in variants:
            s = [r['summary'] for r in results[name][threads]]
            per_run = [x['mean_s'] for x in s]
            run_means[name] = per_run
            lines.append(
                f'| {threads} | {name} | {fmt(statistics.mean(per_run))} | '
                f'{fmt(statistics.mean(x["median_s"] for x in s))} | '
                f'{fmt(statistics.mean(x["stdev_s"] for x in s))} | '
                f'{fmt(statistics.pstdev(per_run) if len(per_run) > 1 else 0.0)} | '
                f'{fmt(statistics.mean(x["total_s"] for x in s))} | '
                f'{fmt(statistics.mean(x["first_step_s"] for x in s))} |'
            )
        base, cand = statistics.mean(run_means['baseline']), statistics.mean(run_means['candidate'])
        ratio = cand / base if base else float('nan')

        # paired per-repeat ratios bound the ratio by run-to-run noise; under --interleave the
        # repeats at a given index ran back to back, so the pairing is meaningful
        paired = [c / b for c, b in zip(run_means['candidate'], run_means['baseline']) if b]
        verdicts.append((threads, ratio, min(paired, default=ratio), max(paired, default=ratio),
                         ratio <= 1.0 + args.tolerance))
    lines.append('')

    lines.append('## Parity verdict\n')
    lines.append('| threads | candidate / baseline | per-repeat range | verdict |')
    lines.append('|---|---|---|---|')
    for threads, ratio, low, high, ok in verdicts:
        lines.append(f'| {threads} | {ratio:.3f} | {low:.3f} - {high:.3f} | {"PASS" if ok else "FAIL"} |')
    lines.append('')
    lines.append(f'The verdict compares the mean of the per-run means against a {args.tolerance:.0%} tolerance. '
                 'A ratio inside the tolerance with a per-repeat range straddling it is noise rather than '
                 'parity; add repeats before reading it either way.\n')

    if len(threads_list) > 1:
        lines.append('## Thread scaling (mean step time at 1 thread / at N threads)\n')
        lines.append('| variant | ' + ' | '.join(f'{t} threads' for t in threads_list) + ' |')
        lines.append('|---|' + '---|' * len(threads_list))
        for name in variants:
            base = statistics.mean(r['summary']['mean_s'] for r in results[name][threads_list[0]])
            scaling = []
            for t in threads_list:
                m = statistics.mean(r['summary']['mean_s'] for r in results[name][t])
                scaling.append(f'{base / m:.2f}x')
            lines.append(f'| {name} | ' + ' | '.join(scaling) + ' |')
        lines.append('')

    lines.append('## Output comparison (final state after the run)\n')
    lines.append(textwrap.dedent('''\
        Three signals, in order of how much they tell you:

        1. A variant that differs from **itself** between identical repeats has a data race. Both code
           paths are deterministic, so repeated runs of one variant must agree bit for bit.
        2. Non-finite values in either variant are a defect regardless of what changed.
        3. Candidate against baseline differs whenever the candidate intentionally changes model
           behavior, so it is only readable against a statement of what the change was meant to do.
           Record that with `--note`; the refs alone do not say.
    '''))
    for threads in threads_list:
        lines.append(f'### {threads} thread(s)\n')
        rd = work / 'runs'
        pairs = [
            ('baseline repeat 0 vs repeat 1', rd / 'baseline' / f't{threads}' / 'run0.npz', rd / 'baseline' / f't{threads}' / 'run1.npz'),
            ('candidate repeat 0 vs repeat 1', rd / 'candidate' / f't{threads}' / 'run0.npz', rd / 'candidate' / f't{threads}' / 'run1.npz'),
            ('candidate vs baseline (repeat 0)', rd / 'candidate' / f't{threads}' / 'run0.npz', rd / 'baseline' / f't{threads}' / 'run0.npz'),
        ]
        for label, a, b in pairs:
            if not (a.exists() and b.exists()):
                continue
            cmp = compare_arrays(a, b)
            lines.append(f'**{label}**\n')
            if 'note' in cmp:
                lines.append(cmp['note'] + '\n')
                continue
            lines.append('| array | max abs diff | max rel diff | cells differing | non-finite (a / b) |')
            lines.append('|---|---|---|---|---|')
            for key, c in cmp.items():
                if 'shape_a' in c:
                    lines.append(f'| {key} | shape mismatch {c["shape_a"]} vs {c["shape_b"]} | | | |')
                else:
                    lines.append(f'| {key} | {c["max_abs"]:.3e} | {c["max_rel"]:.3e} | {c["n_differing"]} | {c["n_nonfinite_a"]} / {c["n_nonfinite_b"]} |')
            lines.append('')

    lines.append('## Notes\n')
    lines.append(textwrap.dedent('''\
        - Per-step times are end to end and include daily output writing and monthly demand loading.
          Both variants do that same work, so the comparison is fair, but a change confined to one
          kernel shows up diluted here: the measured ratio is nearer 1.0 than the kernel's own ratio.
        - Only code paths the sample input and this config exercise are measured. A path that is off
          in both variants, or that exists on only one of them, has no number here.
    ''').rstrip('\n'))
    empty = sorted({
        name for per_thread in results.values() for runs in per_thread.values()
        for r in runs for name in r.get('empty_arrays', ())
    })
    if empty:
        lines.append(f'- Tracked but empty in at least one run, so not compared: {", ".join(empty)}.')
    for note in args.note:
        lines.append(f'- {note}')
    lines.append('')

    report = work / 'parity_report.md'
    report.write_text('\n'.join(lines))
    (work / 'parity_results.json').write_text(json.dumps(
        {name: {str(t): [{k: v for k, v in r.items() if k != 'step_seconds'} | {'summary': r['summary']} for r in runs]
                for t, runs in per_thread.items()} for name, per_thread in results.items()},
        indent=2,
    ))
    print('\n' + '\n'.join(lines))
    print(f'\nReport written to {report}')

    if not args.keep:
        for name, v in variants.items():
            git(repo, 'worktree', 'remove', '--force', str(v['worktree']), capture=False)
        print('Worktrees removed (venvs, data and runs kept under the work dir).')


if __name__ == '__main__':
    main()
