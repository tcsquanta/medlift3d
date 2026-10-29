import glob, os, re, json, zipfile, collections
import numpy as np

ROOTS = ['/kaggle/input', '/kaggle/working']
def find_dir(name):
    hits = [d for r in ROOTS for d in glob.glob(f'{r}/**/{name}', recursive=True) if os.path.isdir(d)]
    return sorted(hits, key=lambda d: (d.count('/'), d))[0] if hits else None
out, logs, work = find_dir('outputs'), find_dir('logs'), find_dir('work')
print('outputs:', out, '\nlogs:   ', logs, '\nwork:   ', work)
ZIP = '/kaggle/working/medlift_report_package.zip'
z = zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED)
MAX_MB, per_nodule = 30, collections.Counter()
def add(path, arc):
    if os.path.getsize(path) <= MAX_MB * 2**20: z.write(path, arc)

# 1) print every results table (md) and add tables, csv, json from outputs
for f in sorted(glob.glob(f'{out}/**/table_*.md', recursive=True)):
    print('\n=====', os.path.relpath(f, out)); print(open(f).read())
for ext in ('md', 'csv', 'json', 'jsonl', 'txt', 'html'):
    for f in glob.glob(f'{out}/**/*.{ext}', recursive=True):
        add(f, 'outputs/' + os.path.relpath(f, out))

# 2) all figures (per-nodule figures: at most 12 of each kind, to keep the zip small)
for f in sorted(glob.glob(f'{out}/**/*.png', recursive=True)):
    rel = os.path.relpath(f, out)
    kind = re.sub(r'LIDC-IDRI-\d+[^/]*?_n\d+', 'NODULE', rel)
    if kind != rel:
        per_nodule[kind] += 1
        if per_nodule[kind] > 12: continue
    add(f, 'outputs/' + rel)

# 3) small result data from work/ (no models, no CT volumes)
for pat in ['**/*.jsonl', '**/*.csv', '**/splits.json', '**/conformal.json', '**/cps_calibration.json',
            '**/verification.json', '**/candidates.json', '**/detections.json', '**/*.txt']:
    for f in glob.glob(f'{work}/{pat}', recursive=True) if work else []:
        add(f, 'work/' + os.path.relpath(f, work))

# 4) logs, plus every PASS / FAIL / INFO line in one file
lines = []
for f in sorted(glob.glob(f'{logs}/*.log')) if logs else []:
    add(f, 'logs/' + os.path.basename(f))
    for l in open(f, errors='ignore'):
        if re.search(r'\[(PASS|FAIL|INFO|fit|val)\]|ALL PASS|FAILED|=====', l):
            lines.append(f'{os.path.basename(f)}: {l.rstrip()}')
z.writestr('checks_summary.txt', '\n'.join(lines))
print('\n===== FAILED CHECKS'); [print(l) for l in lines if '[FAIL]' in l or 'FAILED' in l]

# 5) key numbers computed directly from the per-nodule results
rows = [json.loads(l) for f in glob.glob(f'{work}/**/pipeline.jsonl', recursive=True) for l in open(f) if l.strip()] if work else []
rep = [f'nodules measured: {len(rows)}', 'per split: ' + str(collections.Counter(r.get("split") for r in rows))]
methods = sorted({m for r in rows for m in r.get('methods', {})})
for split in ('cal', 'test'):
    rs = [r for r in rows if r.get('split') == split]
    for m in methods:
        res = {}
        for q in ('volume_mm3', 'mass_mg'):
            p = np.array([r['methods'][m][q] for r in rs if m in r['methods']], float)
            t = np.array([r['gt'][q] for r in rs if m in r['methods']], float)
            if len(p) == 0: continue
            ok = p > 1
            ape = np.abs(p[ok] - t[ok]) / t[ok] * 100
            res[q] = (f'MdAPE {np.median(ape):.1f}%  median signed {np.median((p[ok]-t[ok])/t[ok]*100):+.1f}%' if ok.any() else 'all failed') + f'  failed {100*(1-ok.mean()):.1f}%  n={len(p)}'
        rep.append(f'[{split}] {m:14s} volume: {res.get("volume_mm3")}\n{"":22s}mass:   {res.get("mass_mg")}')
print('\n===== KEY NUMBERS\n' + '\n'.join(rep))
z.writestr('key_numbers.txt', '\n'.join(rep))

# 6) extra figures for the report (each one optional)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
def save(fig, name):
    p = f'/kaggle/working/{name}'; fig.savefig(p, dpi=200, bbox_inches='tight'); plt.close(fig); z.write(p, 'extra_figures/' + name); print('extra figure:', name)
try:   # diffusion training curve
    import pandas as pd
    f = [p for p in glob.glob(f'{work}/**/train_log.csv', recursive=True) if 'detector' not in p and '_old' not in p][0]
    d = pd.read_csv(f); fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(d['step'], d['loss'], lw=0.6, alpha=0.4); ax[0].plot(d['step'], d['loss'].rolling(50, min_periods=1).mean(), lw=1.5)
    ax[0].set(xlabel='step', ylabel='v-prediction loss', yscale='log', title='diffusion training loss')
    for c in [c for c in d.columns if 'psnr' in c.lower()]:
        s = d[['step', c]].dropna(); ax[1].plot(s['step'], s[c], 'o-', ms=3, label=c)
    ax[1].set(xlabel='step', ylabel='PSNR (dB)', title='validation PSNR'); ax[1].legend(fontsize=7)
    save(fig, 'diffusion_training.png')
except Exception as e: print('skip diffusion_training:', e)
try:   # detector training curve
    t = open(f'{logs}/14_detector.log', errors='ignore').read()
    st = [(int(a), float(b)) for a, b in re.findall(r'step (\d+)/\d+\s+loss ([\d.]+)', t)]
    va = [(int(a), float(b)) for a, b in re.findall(r'\[val\] step (\d+): CPM ([\d.]+)', t)]
    fig, ax = plt.subplots(figsize=(6, 3.5)); ax.plot(*zip(*st), lw=1, label='training loss'); ax.set(xlabel='step', ylabel='loss')
    ax2 = ax.twinx(); ax2.plot(*zip(*va), 'o-', color='C1', label='validation CPM'); ax2.set_ylabel('CPM (cal patients)')
    fig.legend(loc='upper center', ncol=2, fontsize=8); save(fig, 'detector_training.png')
except Exception as e: print('skip detector_training:', e)
try:   # dataset: nodule size distribution per split
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for s in ('train', 'cal', 'test'):
        v = [r['gt']['volume_mm3'] for r in rows if r.get('split') == s]
        dm = [r.get('diameter_mm') or r['gt'].get('eq_diameter_mm') for r in rows if r.get('split') == s]
        if v: ax[0].hist(np.log10(v), bins=25, alpha=0.5, label=f'{s} (n={len(v)})'); ax[1].hist([x for x in dm if x], bins=25, alpha=0.5, label=s)
    ax[0].set(xlabel='log10 consensus volume (mm$^3$)', ylabel='nodules'); ax[1].set(xlabel='diameter (mm)'); ax[0].legend(fontsize=8)
    save(fig, 'dataset_nodule_sizes.png')
except Exception as e: print('skip dataset_nodule_sizes:', e)
try:   # H1: mass vs volume error per method, and error vs nodule size
    rs = [r for r in rows if r.get('split') == 'test']
    fig, ax = plt.subplots(1, 2, figsize=(11, 4)); data, labs = [], []
    for m in methods:
        for q, tag in (('volume_mm3', 'V'), ('mass_mg', 'M')):
            p = np.array([r['methods'][m][q] for r in rs if m in r['methods']], float); t = np.array([r['gt'][q] for r in rs if m in r['methods']], float)
            ok = p > 1
            if ok.any(): data.append(np.abs(p[ok] - t[ok]) / t[ok] * 100); labs.append(f'{m}\n{tag}')
    ax[0].boxplot(data, showfliers=False); ax[0].set_xticks(range(1, len(labs) + 1), labs, fontsize=6, rotation=45)
    ax[0].set(ylabel='absolute error (%)', title='volume (V) vs mass (M) error, test nodules')
    for m in methods:
        pr = [(r['gt']['volume_mm3'], abs(np.log(r['methods'][m]['volume_mm3'] / r['gt']['volume_mm3']))) for r in rs if m in r['methods'] and r['methods'][m]['volume_mm3'] > 1]
        if pr: ax[1].scatter(*zip(*pr), s=8, alpha=0.6, label=m)
    ax[1].set(xscale='log', xlabel='consensus volume (mm$^3$)', ylabel='|log(V / V$_{true}$)|', title='volume error vs nodule size'); ax[1].legend(fontsize=7)
    save(fig, 'error_mass_vs_volume.png')
except Exception as e: print('skip error_mass_vs_volume:', e)

z.close()
print(f'\nDONE: {ZIP} ({os.path.getsize(ZIP)/2**20:.1f} MB, {len(zipfile.ZipFile(ZIP).namelist())} files). Download it from the Output panel and upload it here.')
