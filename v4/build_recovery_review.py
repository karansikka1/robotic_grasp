"""Review the varied disturbance dataset, including gripper command traces."""

import argparse
import json
import os
from pathlib import Path

import h5py

from v4.build_robustness_review import TEMPLATE
from v4.collect_recovery import ROOT
from v4.recovery_teacher import CATEGORIES


DESCRIPTIONS = {
    'clean': ('Clean', 'The original teacher executes a complete stack without an injected disturbance.'),
    'gaussian_burst': ('Gaussian bursts', 'One bounded XY Gaussian burst. Object, approach/descent/lift/transfer phase, duration, magnitude, and sample-hold time vary across trajectories.'),
    'drop_block': ('Actual block drops', 'The gripper opens while lifting the selected block. The supervisor retreats, waits for landing, and retries the same object. Blue recovery preserves the green-red base.'),
    'sideways_error': ('Sideways errors', 'A brief directional XY command bias during lift or transfer. Direction, magnitude, duration, object, and timing vary.'),
    'gripper_interrupt': ('Gripper interruptions', 'Near a pick, briefly delay closure or reopen just after contact. Intended labels remain the supervisor commands; executed gripper commands include the interruption.'),
}


def build(root=ROOT):
    root = Path(root).resolve()
    summary = json.loads((root / 'summary.json').read_text())
    video_index = root / 'videos.json'
    if video_index.exists():
        entries = json.loads(video_index.read_text())['videos']
    else:
        entries = []
        for path in sorted((root / 'jobs').glob('*.json')):
            state = json.loads(path.read_text())
            r = state['result']
            if r is not None and r['video']:
                entries.append({'path': r['trajectory'], 'seed': r['seed'], 'steps': r['steps'],
                                'category': r['category'], 'config': r['config'],
                                'partition': state['slot']['partition'], 'video': r['video'],
                                'event': r['event'], 'recoveries': r['recoveries']})
    records = []
    for entry in entries:
        with h5py.File(entry['path'], 'r') as f:
            records.append({**entry, 'video': os.path.relpath(entry['video'], root),
                            'path': os.path.relpath(entry['path'], root),
                            'actions': f['actions'][:].round(6).tolist(),
                            'executed': f['executed_actions'][:].round(6).tolist(),
                            'stages': f['stage'].asstr()[:].tolist(),
                            'active': f['perturbation/active'][:].tolist(),
                            'changed': f['perturbation/changed'][:].tolist(),
                            'events': f['perturbation/event_id'][:].tolist(),
                            'recovery_active': f['perturbation/recovery_active'][:].tolist(),
                            'changed_steps': int(f['perturbation/changed'][:].sum())})
    records.sort(key=lambda r: (CATEGORIES.index(r['category']), r['config']['target_phase'], r['seed']))
    compatible_summary = {**summary, 'categories': {c: {**v, 'validation_trajectories': v['validation']}
                                                   for c, v in summary['categories'].items()}}
    data = {'summary': compatible_summary,
            'categories': [{'id': c, 'name': DESCRIPTIONS[c][0], 'description': DESCRIPTIONS[c][1]} for c in CATEGORIES],
            'videos': records}
    page = TEMPLATE.replace('__DATA__', json.dumps(data, separators=(',', ':')).replace('<', '\\u003c'))
    page = page.replace('Trajectory review', 'Recovery demonstration review')
    page = page.replace('<title>BC trajectory review</title>', '<title>Recovery demonstration review</title>')
    page = page.replace('Rotation and gripper commands are unchanged in every category. Both cameras are shown.',
                        'Gripper faults delay closure or reopen a grasp; drop events physically release a block. Rotation commands are unchanged. Both cameras are shown.')
    page = page.replace('aspect-ratio:2/1', 'aspect-ratio:1.6')
    page = page.replace('height:210px', 'height:270px').replace('h=210', 'h=270')
    page = page.replace('row=(bottom-top)/3', 'row=(bottom-top)/4')
    page = page.replace('for(let axis=0;axis<3;axis++)', 'for(let axis=0;axis<4;axis++)')
    page = page.replace("['X','Y','Z'][axis]", "['X','Y','Z','G'][axis]")
    page = page.replace('a[axis]', 'a[[0,1,2,6][axis]]').replace('(row-14)/1.5', '(row-14)/2.2')
    page = page.replace('Movement commands', 'Movement and gripper commands')
    page = page.replace('XYZ commands over time', 'XYZ and gripper commands over time')
    page = page.replace('not measured displacement.', 'not measured displacement. G is the gripper: −1 opens, +1 closes.')
    page = page.replace('<option value="0.5">', '<option value="0.5" selected>')
    page = page.replace('<option value="1" selected>', '<option value="1">')
    page = page.replace('Next disturbance</button>', 'Jump to disturbance</button>')
    page = page.replace("el('event').disabled=!['dropped_commands','sustained_bias'].includes(v.category)",
                        "el('event').disabled=!v.event")
    start = page.index("el('event').onclick=")
    end = page.index('\nfunction update()', start)
    page = page[:start] + "el('event').onclick=()=>{if(current&&current.event)seek(Math.max(0,current.event.start-10))};" + page[end:]
    page = page.replace("el('description').textContent=c.description;",
                        "el('description').textContent=c.description+(v.event?' This example: '+v.event.target_object+', '+v.config.trigger.replaceAll('_',' ')+', event '+(v.event.start/20).toFixed(2)+'–'+(v.event.end/20).toFixed(2)+' s.':'');")
    page = page.replace("' mm';draw(s)", "' mm · gripper target/applied: '+current.actions[s][6].toFixed(0)+' / '+current.executed[s][6].toFixed(0);draw(s)")
    page = page.replace("current.changed[s]?'Movement command disturbed'", "current.changed[s]?'Command disturbed'")
    page = page.replace("'Teacher command executed unchanged';el('disturbance')", "current.recovery_active[s]?'Retreat and retry after lost contact':'Teacher command executed unchanged';el('disturbance')")
    page = page.replace('<body><main>', '<body><main><p class="hint"><a href="../">Earlier mild dataset</a> · <a href="../visible_pilots/">Three approved previews</a></p>')
    (root / 'index.html').write_text(page)
    return {'page': str(root / 'index.html'), 'videos': len(records),
            'saved_trajectories': summary['successful_trajectories']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(build(args.root), indent=2))


if __name__ == '__main__':
    main()
