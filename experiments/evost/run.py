"""Run a preserved EvoST experiment entry point. See README.md in this directory."""
from pathlib import Path
import runpy
import sys

ENTRIES = {
    'maintenance': ('parameter_groups', 'e15_maintain.py'),
    'prompt': ('prompt_and_scratch', 'e23_eac.py'),
    'arrival': ('sensor_arrival', 'e21_handover2.py'),
    'age': ('checkpoint_ages', 'e11_ages.py'),
    'stid': ('stid', 'e13_stid.py'),
    'gwn': ('graph_wavenet', 'e14_gwn.py'),
    'budget': ('history_budget', 'e16_budget.py'),
}

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print('Usage: python experiments/evost/run.py STUDY [experiment arguments]')
        print('Studies: ' + ', '.join(ENTRIES))
        print('Use STUDY --help for experiment arguments.')
        return
    name = sys.argv[1]
    if name not in ENTRIES:
        raise SystemExit('Unknown study: ' + name)
    group, filename = ENTRIES[name]
    source = Path(__file__).resolve().parent / group
    sys.path.insert(0, str(source))
    sys.argv = [str(source / filename)] + sys.argv[2:]
    runpy.run_path(sys.argv[0], run_name='__main__')

if __name__ == '__main__':
    main()
