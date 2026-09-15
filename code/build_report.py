"""Build a single offline HTML report, including media and a downloadable code bundle.

Run: python -m pip install -r requirements_report.txt && python build_report.py
"""
import base64
import hashlib
import html
import io
import json
import mimetypes
from pathlib import Path
import re
import textwrap
from urllib.parse import unquote

import av
from bs4 import BeautifulSoup
import markdown
from markdown.extensions.toc import slugify

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'solution.html'


def document_id(path):
    return 'doc-' + slugify(path.as_posix(), '-')


def architecture(source):
    labels = dict(re.findall(r'(\w+)\[([^\]]+)\]', source))
    edges = re.findall(r'(\w+)(?:\[[^\]]+\])?\s*(-->|-\.->)\s*(\w+)', source)
    positions = {'A': (250, 40), 'B': (250, 125), 'C': (250, 210), 'D': (250, 295),
                 'P': (575, 295), 'E': (250, 380), 'F': (250, 465), 'G': (575, 465)}
    expected = {('A', 'B'), ('B', 'C'), ('C', 'D'), ('P', 'D'), ('D', 'E'), ('E', 'F'), ('E', 'G')}
    if set(labels) != set(positions) or {(a, b) for a, _, b in edges} != expected:
        raise ValueError('Architecture changed; update the static diagram layout before rebuilding')
    parts = ['<div class="diagram"><svg viewBox="0 0 720 525" role="img" aria-labelledby="architecture-title architecture-desc" xmlns="http://www.w3.org/2000/svg">',
             '<title id="architecture-title">Visual policy architecture</title>',
             '<desc id="architecture-desc">Both camera views pass through ResNet18 and spatial features to an adapter. Robot measurements also enter the adapter. The LSTM produces actions and a separate stage prediction.</desc>',
             '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#47796a"/></marker></defs>']
    for a, style, b in edges:
        x, y = positions[a]; xx, yy = positions[b]
        if a == 'P': path = f'M {x-120} {y} H {xx+120}'
        elif b == 'G': path = f'M {x+120} {y} H {xx} V {yy-29}'
        else: path = f'M {x} {y+29} V {yy-29}'
        dash = ' stroke-dasharray="5 5"' if style == '-.->' else ''
        parts.append(f'<path d="{path}" fill="none" stroke="#47796a" stroke-width="2"{dash} marker-end="url(#arrow)"/>')
    for key, (x, y) in positions.items():
        parts.append(f'<rect x="{x-120}" y="{y-28}" width="240" height="56" rx="8" fill="white" stroke="#96b5a6"/>')
        lines = textwrap.wrap(labels[key], 31)
        for i, line in enumerate(lines):
            parts.append(f'<text x="{x}" y="{y+5+(i-(len(lines)-1)/2)*17}" text-anchor="middle" fill="#234d43" font-family="system-ui,sans-serif" font-size="12">{html.escape(line)}</text>')
    return ''.join(parts) + '</svg></div>'


def build():
    files = {}
    for p in sorted(ROOT.rglob('*')):
        rel = p.relative_to(ROOT)
        if not p.is_file() or any(part in ('outputs', '__pycache__', '.venv', '.git') for part in rel.parts):
            continue
        if p.suffix in ('.html', '.pyc') or (p.name.startswith('.') and p.name != '.gitignore'):
            continue
        data = p.read_bytes()
        files[rel.as_posix()] = {'mime': mimetypes.guess_type(p.name)[0] or 'application/octet-stream',
                                'data': base64.b64encode(data).decode(), 'sha256': hashlib.sha256(data).hexdigest()}
    documents = [Path('solution.md')] + [p.relative_to(ROOT) for p in sorted(ROOT.glob('*.md')) if p.name != 'solution.md']
    videos, posters, video_sizes = {}, {}, {}
    def poster(name):
        if name not in posters:
            with av.open(str(ROOT / name)) as container:
                frame = next(container.decode(video=0)).to_image()
                video_sizes[name] = frame.size
            out = io.BytesIO(); frame.save(out, format='JPEG', quality=80)
            posters[name] = 'data:image/jpeg;base64,' + base64.b64encode(out.getvalue()).decode()
        return posters[name]
    rendered = {}
    for rel in documents:
        source = (ROOT / rel).read_text()
        # Display-only copy edits; source Markdown remains the editable original.
        if rel.name == 'solution.md':
            source = source.replace('<to be updated>\n', '').replace('### TLDR', '### Summary')
            source = source.replace('variantions', 'variations').replace('previliged', 'privileged')
            source = source.replace('a MLP', 'an MLP').replace('constraint on not adding additional dependencies', 'the restriction on adding dependencies')
            source = source.replace('RL on uninitialized policy did not work well even with curriculum learning', 'RL from an uninitialized policy did not work well, even with curriculum learning.')
            source = source.replace('BC was done by generating episodes with the simulator as the teacher with some variations.', 'For BC, I generated demonstrations using a scripted teacher in the simulator, including variations and recovery episodes.')
            source = source.replace('Things that helped to train the model -> (1) predicting a high-level state for the current episode, (2) pre-training resnet with localization information', 'Two changes helped: predicting the current task stage and pre-training ResNet with localization information.')
            source = source.replace('Inline playback depends on the Markdown viewer; use the links if video elements are not supported.\n', '')
            source = re.sub(r'^## Results\s*(?=##)', '', source, flags=re.M)
        soup = BeautifulSoup(markdown.markdown(source, extensions=['tables', 'fenced_code', 'toc', 'sane_lists']), 'html.parser')
        for pre in soup.select('pre:has(code.language-mermaid)'):
            pre.replace_with(BeautifulSoup(architecture(pre.code.get_text()), 'html.parser'))
        if rel.name == 'solution.md':
            for p in list(soup.find_all('p')):
                if len(p.contents) == 1 and p.strong and p.get_text() in ('Non-visual Policy:', 'Correction Data in BC:', 'Visual policy:'):
                    title = p.get_text().rstrip(':'); h = soup.new_tag('h3')
                    h.string = title; h['id'] = slugify(title, '-'); p.replace_with(h)
            for h in soup.find_all('h3'):
                if h.get_text() not in ('Non-visual Policy', 'Correction Data in BC', 'Visual policy'):
                    h.name = 'h2'
        for tag in soup.select('[id]'):
            if tag.name not in ('svg', 'title', 'desc', 'marker'):
                tag['id'] = document_id(rel) + '--' + tag['id']
        for img in soup.find_all('img'):
            name = (rel.parent / img['src']).as_posix()
            img['src'] = 'data:' + files[name]['mime'] + ';base64,' + files[name]['data']
        for i, video in enumerate(soup.find_all('video')):
            name = (rel.parent / video['src']).as_posix()
            vid = document_id(rel) + '-video-' + str(i)
            video['id'] = vid; video['data-asset'] = name; del video['src']
            video['poster'] = poster(name); video['preload'] = 'metadata'; video['playsinline'] = ''
            width, height = video_sizes[name]
            video['style'] = f'aspect-ratio: {width} / {height}'
            video['aria-label'] = name.rsplit('/', 1)[-1].replace('_', ' ')
            videos.setdefault(name, vid)
        for table in soup.find_all('table'):
            if table.find('video'):
                table['class'] = ['video-table']
                if len(table.select('thead tr th')) == 2:
                    table['class'].append('video-pair')
            if table.find('th') and table.find('th').get_text() == 'Rank': table['class'] = 'ranked'
            for th in table.find_all('th'): th['scope'] = 'col'
            wrapper = soup.new_tag('div', attrs={'class': 'table-scroll', 'tabindex': '0', 'role': 'region', 'aria-label': 'Scrollable data table'})
            table.wrap(wrapper)
        rendered[rel] = soup
    source_sections = []
    for rel, soup in rendered.items():
        for a in soup.find_all('a', href=True):
            href = unquote(a['href'])
            if re.match(r'https?://', href):
                raise ValueError(f'External link must be resolved for offline report: {href}')
            path, _, anchor = href.partition('#')
            target = rel.parent / path if path else rel
            if target in documents:
                a['href'] = '#' + document_id(target) + ('--' + anchor if anchor else '')
            elif target.suffix == '.html' and target.name == 'solution.html':
                a['href'] = '#doc-solutionmd'
            elif target.suffix == '.mp4':
                a['href'] = '#' + videos[target.as_posix()]; a['data-video'] = videos[target.as_posix()]
            elif target.as_posix() in files:
                name = target.as_posix()
                if target.suffix in ('.py', '.txt', '.json', '.css', '.js'):
                    a['href'] = '#source-' + slugify(name, '-')
                    if name not in source_sections: source_sections.append(name)
                else:
                    a['href'] = '#downloads'; a['data-download'] = name
            else:
                raise ValueError(f'Unresolved local link in {rel}: {href}')
    main = rendered[Path('solution.md')]
    nav = ''.join(f'<a class="{"sub" if h.name=="h3" else ""}" href="#{h["id"]}">{html.escape(h.get_text())}</a>' for h in main.find_all(['h2', 'h3']))
    supporting = []
    for rel in documents[1:]:
        title = rendered[rel].find(re.compile('^h[1-6]$'))
        label = title.get_text() if title else str(rel)
        supporting.append(f'<details id="{document_id(rel)}"><summary>{html.escape(label)}</summary><div class="appendix-body">{rendered[rel]}</div></details>')
    sources = []
    for name in source_sections:
        sources.append(f'<details id="source-{slugify(name,"-")}"><summary>{html.escape(name)}</summary><p class="source-download"><a href="#downloads" data-download="{name}">Download file</a></p><pre><code>{html.escape((ROOT/name).read_text())}</code></pre></details>')
    css = (ROOT / 'report_assets/report.css').read_text()
    js = (ROOT / 'report_assets/report.js').read_text()
    embedded = json.dumps(files, separators=(',', ':'))
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; media-src data: blob:; connect-src 'none'; font-src 'none'; object-src 'none'; base-uri 'none'">
<title>Learning to stack from demonstrations — BC report</title><style>{css}</style></head>
<body><div class="layout"><nav aria-label="Report navigation"><div class="brand">Robotic stacking / BC</div><div class="links">{nav}<hr><a href="#supporting">Supporting notes</a><a href="#sources">Source code</a></div><div class="nav-actions" id="downloads"><button id="download-bundle">Download code bundle</button><button id="print-report">Print report</button></div></nav>
<main class="content" id="top"><header><div class="eyebrow">Take-home report</div><h1>Learning to stack<br>from demonstrations</h1><p class="subtitle">Behavior cloning, visual localization, and evaluation for a two-camera robotic stacking policy.</p></header>
<noscript><p class="noscript">Enable JavaScript for embedded video playback and file downloads. The report text and figures are available below.</p></noscript>
<article id="{document_id(Path('solution.md'))}">{main}</article>
<section class="appendices" id="supporting"><h2>Supporting notes</h2><p>Training, evaluation, and setup details referenced in the report.</p>{''.join(supporting)}</section>
<section class="source-list" id="sources"><h2>Referenced source files</h2>{''.join(sources)}</section>
<footer>This file includes the report, supporting notes, images, videos, and downloadable code and checkpoint. It opens offline. The code bundle contains the original Markdown and the report builder.</footer><a class="top-link" href="#top">Back to top ↑</a></main></div>
<script id="embedded-files" type="application/json">{embedded}</script><script>{js}</script></body></html>'''
    final = BeautifulSoup(page, 'html.parser')
    ids = [t['id'] for t in final.select('[id]')]
    if len(ids) != len(set(ids)): raise ValueError('Duplicate HTML IDs')
    for a in final.select('a[href]'):
        if a['href'].startswith('#') and a['href'][1:] not in ids:
            raise ValueError('Broken anchor: ' + a['href'])
    OUT.write_text(page)
    print(f'Built {OUT.name}: {OUT.stat().st_size/1024/1024:.1f} MiB; {len(files)} bundled files, {len(videos)} unique videos, {len(documents)} documents.')


if __name__ == '__main__':
    build()
