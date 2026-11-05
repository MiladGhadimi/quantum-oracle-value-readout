"""Regenerate Fig. 4 from saved ensemble statistics without rerunning solvers."""
import json
from pathlib import Path
from reproduce import plot_scan


if __name__ == '__main__':
    results = json.loads((Path(__file__).resolve().parent / 'results.json').read_text())
    plot_scan(results['mesh'])
    print('Regenerated sigma_scan_expectation.pdf from saved results.')
