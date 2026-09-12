"""Build a three-example video page with direct event and recovery controls."""

import argparse
import html
import json
import os
from pathlib import Path

from v4.visible_perturbation_pilots import ROOT


INFO = {
    'drop_block': ('1', 'Drop the block', 'The gripper opens while holding green. The block falls onto the table; an added recovery controller clears the gripper, waits for the block to land, and picks it up again.'),
    'gaussian_burst': ('2', 'A visible Gaussian burst', 'One 1-second burst of bounded XY Gaussian noise near the first approach. Each sample lasts 0.25 seconds so movement errors do not immediately cancel. The teacher then resumes.'),
    'sideways_error': ('3', 'Move sideways while holding', 'One 0.5-second sideways command bias while holding green. The arm moves away from its hold position, then the teacher resumes the lift and stack.'),
}


def build(root):
    root = Path(root).resolve()
    data = json.loads((root / 'examples.json').read_text())
    cards = []
    for category in INFO:
        result = next(r for r in data['examples'] if r['category'] == category)
        number, title, description = INFO[category]
        video = html.escape(os.path.relpath(result['video'], root), quote=True)
        event = result['event_start'] / 20
        end = result['event_end'] / 20
        regrasp = result.get('regrasp_step')
        if category == 'drop_block':
            measure = f"{result['green_downward_travel_cm']:.1f} cm fall"
            explanation = f"Grasp lost during release; grasp restored at {regrasp / 20:.2f} s." if regrasp is not None else 'No subsequent regrasp was recorded.'
        elif category == 'gaussian_burst':
            measure = f"{result['eef_event_displacement_cm']:.1f} cm movement"
            explanation = 'Maximum measured gripper displacement from its position at burst start.'
        else:
            measure = f"{result['green_event_displacement_cm']:.1f} cm movement"
            explanation = 'Maximum measured block displacement from its position at disturbance start.'
        outcome = 'Stack completed' if result['success'] else 'Did not complete: ' + str(result['failure'])
        cards.append(f'''<section id="{category}" class="card" data-start="{event}" data-end="{end}" data-regrasp="{regrasp / 20 if regrasp is not None else end}" data-steps="{result['steps']}">
<div class="film"><video controls playsinline preload="metadata" src="{video}"></video><div class="controls"><button class="jump primary">▶ Play disturbance</button><button class="resume">Watch recovery</button><button class="begin">From start</button><select aria-label="Playback speed"><option value="0.25">0.25× speed</option><option value="0.5" selected>0.5× speed</option><option value="1">1× speed</option></select></div><div class="scrub"><button class="back" aria-label="Previous frame">← Frame</button><span class="clock">0.00 s</span><button class="forward" aria-label="Next frame">Frame →</button></div></div>
<div class="detail"><div class="number">EXAMPLE {number}</div><h2>{title}</h2><p>{description}</p><div class="measure">{measure}</div><p class="small">{explanation}</p><dl><div><dt>Disturbance</dt><dd>{event:.2f}–{end:.2f} s</dd></div><div><dt>Full trajectory</dt><dd>{result['seconds']:.2f} s</dd></div><div><dt>Outcome</dt><dd>{outcome}</dd></div></dl><p class="state" aria-live="polite"></p><a href="{video}" download>Download annotated video ↗</a></div></section>''')
    page = TEMPLATE.replace('__CARDS__', '\n'.join(cards))
    (root / 'index.html').write_text(page)
    return root / 'index.html'


TEMPLATE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Three disturbance previews</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#10151d;color:#edf3f8;font:15px/1.55 system-ui,sans-serif}main{max-width:1260px;margin:auto;padding:40px 28px 60px}h1{font-size:38px;font-weight:650;letter-spacing:-1.2px;margin:8px 0 12px;line-height:1.15}h2{font-size:24px;line-height:1.2;margin:8px 0 14px}p{color:#b4c2ce;margin:10px 0}.eyebrow,.number{font-size:11px;letter-spacing:2px;color:#78d9e5}.intro{max-width:850px}.links{display:flex;gap:12px;margin:24px 0 32px;flex-wrap:wrap}a{color:#81dce7}.links a{padding:8px 14px;border:1px solid #344655;border-radius:22px;text-decoration:none}.card{display:grid;grid-template-columns:minmax(0,1fr) 325px;background:#18212c;border:1px solid #32404e;border-radius:16px;overflow:hidden;margin:22px 0 36px;scroll-margin-top:20px}.film{background:#070c12}video{width:100%;display:block;aspect-ratio:1.6;background:#070c12}.detail{padding:25px;border-left:1px solid #32404e}.controls{padding:14px 16px;display:flex;gap:8px;flex-wrap:wrap}button,select{font:inherit;font-size:13px;cursor:pointer;background:#202e3c;color:#ecf4f7;border:1px solid #3b5061;border-radius:8px;padding:9px 12px}button:hover{border-color:#86dbe4}.primary{background:#285660;border-color:#72b8c5}.scrub{display:flex;justify-content:center;align-items:center;gap:18px;padding:0 16px 16px}.scrub button{padding:4px 10px;font-size:12px}.clock{min-width:65px;text-align:center;font-variant-numeric:tabular-nums;color:#b3c8d6}.measure{font-size:28px;color:#ffd097;margin-top:25px}.small{font-size:12px}dl{margin:25px 0}dl div{display:flex;justify-content:space-between;gap:15px;padding:8px 0;border-bottom:1px solid #34424e}dt{color:#aabccb;font-size:12px}dd{margin:0;text-align:right;font-size:13px}.state{color:#f4c58d;min-height:24px;font-size:13px}footer{font-size:13px;border-top:1px solid #32404e;padding-top:20px;color:#adc0cc}footer a{margin-right:20px}button:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid #82dae5;outline-offset:3px}@media(max-width:900px){.card{grid-template-columns:1fr}.detail{border-left:0;border-top:1px solid #32404e}.measure{margin-top:15px}dl{max-width:440px}.detail p{max-width:650px}}@media(max-width:500px){main{padding:25px 14px}h1{font-size:31px}.controls{padding:10px;gap:6px}button,select{font-size:12px;padding:8px}.detail{padding:20px}}
</style></head><body><main><div class="eyebrow">ROBOTIC GRASP / REVIEW BEFORE SCALING</div><h1>Three deliberate disruptions.</h1><div class="intro"><p>A real dropped block, a localized Gaussian burst, and a sideways movement error. All three use the same starting layout, and all three completed the stack.</p><p class="small">Each player starts just before its disturbance. The video banner marks when it is active. Playback defaults to half speed; use “From start” to see the full trajectory. These are privileged, stage-triggered teacher previews with explicit hold/recovery logic. They have not been added to training or validation.</p></div><nav class="links" aria-label="Examples"><a href="#drop_block">1 · Dropped block</a><a href="#gaussian_burst">2 · Gaussian burst</a><a href="#sideways_error">3 · Sideways error</a></nav>
__CARDS__
<footer><a href="examples.json">Exact settings and measurements</a><a href="validation.json">Video and replay checks</a><a href="../">Earlier mild-disturbance dataset</a><p>Three examples only. The next collection will wait for your review.</p></footer></main><script>
'use strict';
for(const card of document.querySelectorAll('.card')){
const video=card.querySelector('video'),start=Number(card.dataset.start),end=Number(card.dataset.end),total=Number(card.dataset.steps),speed=card.querySelector('select');
function update(){const t=video.currentTime;card.querySelector('.clock').textContent=t.toFixed(2)+' s';card.querySelector('.state').textContent=t<start?'Normal teacher → disturbance ahead':t<end?'Disturbance active':'Recovery / normal stacking resumes'}
function cue(time,play=false){video.pause();video.currentTime=Math.max(0,Math.min(total/20,time));video.playbackRate=Number(speed.value);update();if(play)video.play().catch(()=>{})}
video.addEventListener('loadedmetadata',()=>cue(Math.max(0,start-0.75)));
video.addEventListener('timeupdate',update);video.addEventListener('seeked',update);
card.querySelector('.jump').onclick=()=>cue(Math.max(0,start-0.5),true);
card.querySelector('.resume').onclick=()=>cue(end,true);
card.querySelector('.begin').onclick=()=>cue(0,true);
card.querySelector('.back').onclick=()=>cue((Math.round(video.currentTime*20)-1)/20);
card.querySelector('.forward').onclick=()=>cue((Math.round(video.currentTime*20)+1)/20);
speed.onchange=()=>{video.playbackRate=Number(speed.value)};
video.addEventListener('play',()=>{for(const other of document.querySelectorAll('video'))if(other!==video)other.pause()});
video.addEventListener('error',()=>card.querySelector('.state').textContent='Could not load video; use the download link.');
}
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(build(args.root))


if __name__ == '__main__':
    main()
