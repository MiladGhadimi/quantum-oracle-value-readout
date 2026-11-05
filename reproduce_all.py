"""Run the numerical reproduction or rebuild outputs from bundled data."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--from-saved', action='store_true',
        help='Validate bundled matrices and regenerate tables/plots without solving SDPs.'
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    env = os.environ.copy()
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[name] = '1'
    scripts = [] if args.from_saved else ['numerics/reproduce.py']
    scripts += ['numerics/validate.py', 'numerics/make_tables.py',
                'numerics/plot_saved_scan.py', 'figures_source/build_analytic_figures.py']
    for script in scripts:
        print(f'Running {script}', flush=True)
        subprocess.run([sys.executable, str(root / script)],
                       cwd=root, env=env, check=True)
    print('Reproduction completed.', flush=True)


if __name__ == '__main__':
    main()
