#!/usr/bin/env python
"""Step 18 - one HTML page with every result: tables, figures and the PASS / FAIL checks of all steps.

Plain Python 3, no packages: runs on Kaggle, a laptop or the pod, on the repository's outputs/ or on the
outputs/ folder of an unzipped medlift_kaggle.zip / medlift_full_results.zip. The page is a single file
(figures embedded): open it in any browser, also on a phone or tablet, or send it to your supervisor.

    python scripts/18_report.py                                  # outputs/ -> outputs/report.html
    python scripts/18_report.py --outputs ~/Downloads/medlift_full_results/outputs
    python scripts/18_report.py --nodule-figures 20               # more per-nodule pictures
"""
import argparse
import base64
import datetime
import glob
import html
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (folder, title, what to look at, tables (None = all), figure patterns, max figures per pattern)
SECTIONS = [
    ("10_evaluation", "Accuracy and calibrated intervals (H1, H2, H3)",
     "Test patients only. table_accuracy: volume and mass error of every method (MdAPE = median absolute % "
     "error, lower is better). H1: mass error < volume error. H2: ours (Diffusion+Gaussians) beats the "
     "physics-only methods. table_intervals: do the 90% intervals contain the truth ~90% of the time (H3)?",
     ["table_accuracy.md", "table_intervals.md"],
     ["volume_errors.png", "mass_errors.png", "volume_scatter.png", "bland_altman_*.png"], 4),
    ("11_growth", "Growth from two DTS scans (H4)",
     "Virtual follow-up scans with a known doubling time: is the mass doubling time (MDT) more accurate "
     "than the volume doubling time (VDT)?", None, ["*.png"], 6),
    ("14_detector", "Nodule detector",
     "Learned detector vs the training-free baseline on the test patients. CPM: average sensitivity at "
     "1/8 ... 8 false positives per scan (higher is better).", ["table_detection.md"], ["froc_test.png", "loss.png"], 2),
    ("08_diffusion", "Diffusion training",
     "Training loss and validation pictures (DTS input | diffusion sample | true CT) at the first, middle "
     "and last check.", [], ["loss.png", "val_step*.png"], 3),
    ("15_fit", "Under- / overfitting check",
     "Verdict per model from the training curves; held-out vs training losses.", None, ["fit_curves.png"], 1),
    ("16_uncertainty", "Uncertainty quantification (conformal, H5)",
     "Interval variants, coverage per size group, calibrated EUPS category probabilities vs the raw "
     "ensemble vote, all methods. Plots: prediction intervals, regression with intervals, residuals, "
     "Q-Q, interval widths, coverage at every level, predictive distributions.",
     ["table_uq_intervals.md", "table_uq_groups.md", "table_uq_risk.md", "table_uq_methods.md"],
     ["volume_uq_overview.png", "mass_uq_overview.png", "reliability.png", "group_coverage.png",
      "volume_cpd_examples.png"], 2),
    ("17_explain", "Explainability (H6)",
     "Seen / unseen split: what the 15 X-ray views measure vs what the diffusion prior adds (unseen skill "
     "> 0 = recovers the invisible part, < 0 = invents). Error drivers and detector saliency.",
     ["table_explain.md", "table_error_drivers.md", "table_saliency.md"],
     ["*_seen_unseen.png", "saliency_examples.png"], 4),
    ("13_geometry", "Acquisition geometry (views / sweep angle)", "How much the 15-view, 30 deg choice matters.",
     None, ["*.png"], 4),
    ("09_pipeline", "Example nodules: truth vs every method", "Side by side for a few cal / test nodules.",
     [], ["*.png"], 8),
]


def inline(text):
    t = html.escape(text)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*\s][^*]*)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"(?<![\w])_([^_]+)_(?![\w])", r"<i>\1</i>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", t)                 # links: keep the text
    return t


def md_to_html(md):
    """Enough Markdown for our tables and reports: pipe tables, headings, lists, paragraphs."""
    out, para, table, items = [], [], [], []

    def flush():
        if para:
            out.append("<p>" + " ".join(inline(p) for p in para) + "</p>")
            para.clear()
        if items:
            out.append("<ul>" + "".join(f"<li>{inline(i)}</li>" for i in items) + "</ul>")
            items.clear()
        if table:
            rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in table
                    if not re.match(r"^\s*\|?[\s:|-]+\|?\s*$", r)]
            if rows:
                head = "".join(f"<th>{inline(c)}</th>" for c in rows[0])
                body = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
                out.append(f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')
            table.clear()

    for line in md.splitlines():
        s = line.rstrip()
        if s.lstrip().startswith("|"):
            if para or items:
                tmp = list(table)
                table.clear()
                flush()
                table.extend(tmp)
            table.append(s)
            continue
        if table:
            flush()
        if not s.strip():
            flush()
        elif s.startswith("#"):
            flush()
            out.append(f"<h4>{inline(s.lstrip('#').strip())}</h4>")
        elif re.match(r"^\s*[*-] ", s):
            if para:
                p = list(items)
                items.clear()
                flush()
                items.extend(p)
            items.append(re.sub(r"^\s*[*-] ", "", s))
        else:
            if items:
                flush()
            para.append(s.strip())
    flush()
    return "\n".join(out)


def img_tag(path):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    name = os.path.basename(path)
    return (f'<figure><img loading="lazy" src="data:image/png;base64,{data}" alt="{html.escape(name)}">'
            f"<figcaption>{html.escape(name)}</figcaption></figure>")


def pick(paths, n):
    """At most n paths, spread over the sorted list (first, middle, last ...)."""
    paths = sorted(paths)
    if len(paths) <= n:
        return paths
    idx = sorted({round(i * (len(paths) - 1) / (n - 1)) for i in range(n)}) if n > 1 else [len(paths) - 1]
    return [paths[i] for i in idx]


def verification(folder):
    p = os.path.join(folder, "verification.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def status(r):
    return "INFO" if r["ok"] is None else ("PASS" if r["ok"] else "FAIL")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", default=os.path.join(ROOT, "outputs"), help="the outputs/ folder")
    ap.add_argument("--out", default=None, help="default: <outputs>/report.html")
    ap.add_argument("--nodule-figures", type=int, default=None, help="per-nodule pictures (09 / 17)")
    args = ap.parse_args()
    od = os.path.abspath(os.path.expanduser(args.outputs))
    if not os.path.isdir(od):
        raise SystemExit(f"no outputs folder at {od} (unzip first; pass --outputs <path to outputs>)")
    out = args.out or os.path.join(od, "report.html")
    steps = sorted(d for d in os.listdir(od) if os.path.isdir(os.path.join(od, d)) and re.match(r"^\d\d_", d))

    # ---- overview: every step's checks, the hypotheses -------------------------------------------
    ver = {d: verification(os.path.join(od, d)) for d in steps}
    rows, hyp, fails = [], [], []
    for d, v in ver.items():
        if v is None:
            continue
        n_p = sum(r["ok"] is True for r in v["rows"])
        n_f = sum(r["ok"] is False for r in v["rows"])
        rows.append(f'<tr><td>{html.escape(v["title"])}</td><td>{n_p}</td><td class="{"bad" if n_f else "ok"}">{n_f}</td></tr>')
        for r in v["rows"]:
            if re.match(r"^H\d", str(r["check"])) or "H1" in str(r["check"]) or "(H" in str(r["check"]):
                hyp.append(r)
            if r["ok"] is False:
                fails.append((v["title"], r))
    parts = ["<h2>Overview</h2>",
             "<p>Every step checks itself (PASS / FAIL). Hypothesis lines (H1-H6) are scientific results, not "
             "code checks: report them whichever way they come out.</p>"]
    if hyp:
        parts.append("<h3>Hypotheses</h3><div class='tw'><table><thead><tr><th>result</th><th>hypothesis</th>"
                     "<th>value</th></tr></thead><tbody>" + "".join(
                         f'<tr><td class="{"ok" if r["ok"] else ("bad" if r["ok"] is False else "")}">{status(r)}</td>'
                         f'<td>{inline(str(r["check"]))}</td><td>{inline(str(r["value"]))}</td></tr>' for r in hyp)
                     + "</tbody></table></div>")
    parts.append("<h3>Checks per step</h3><div class='tw'><table><thead><tr><th>step</th><th>passed</th>"
                 "<th>failed</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")
    if fails:
        parts.append("<h3>Failed checks</h3><ul>" + "".join(
            f"<li><b>{html.escape(t)}</b>: {inline(str(r['check']))} = {inline(str(r['value']))}</li>" for t, r in fails)
            + "</ul>")

    # ---- sections ----------------------------------------------------------------------------------
    toc = []
    for folder, title, what, tables, patterns, nmax in SECTIONS:
        fd = os.path.join(od, folder)
        if not os.path.isdir(fd):
            continue
        anchor = folder
        toc.append(f'<li><a href="#{anchor}">{html.escape(title)}</a></li>')
        parts.append(f'<h2 id="{anchor}">{html.escape(title)}</h2><p class="what">{html.escape(what)}</p>')
        mds = ([os.path.join(fd, t) for t in tables] if tables is not None else
               sorted(p for p in glob.glob(os.path.join(fd, "*.md")) if not p.endswith("verification.md")))
        for p in mds:
            if os.path.exists(p):
                with open(p, errors="replace") as f:
                    parts.append(f"<h3>{html.escape(os.path.basename(p))}</h3>" + md_to_html(f.read()))
        figs, seen = [], set()
        for pat in patterns:
            n = args.nodule_figures if (args.nodule_figures and folder in ("09_pipeline", "17_explain")
                                        and pat.startswith("*")) else nmax
            for p in pick([q for q in glob.glob(os.path.join(fd, pat)) if q not in seen], n):
                seen.add(p)
                figs.append(p)
        if figs:
            parts.append('<div class="figs">' + "".join(img_tag(p) for p in figs) + "</div>")
        v = ver.get(folder)
        if v:
            parts.append("<details><summary>checks of this step</summary>" + md_to_html(
                open(os.path.join(fd, "verification.md"), errors="replace").read()) + "</details>")
    rest = [d for d in steps if d not in {s[0] for s in SECTIONS} and ver.get(d)]
    if rest:
        toc.append('<li><a href="#other">Other steps (checks)</a></li>')
        parts.append('<h2 id="other">Other steps (checks)</h2>')
        for d in rest:
            parts.append(f"<details><summary>{html.escape(ver[d]['title'])}</summary>" + md_to_html(
                open(os.path.join(od, d, "verification.md"), errors="replace").read()) + "</details>")

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>MedLift-3D results</title>
<style>
:root {{ --bg:#fff; --fg:#1a1a1a; --mut:#555; --line:#d0d7de; --head:#eef2f6; --ok:#1a7f37; --bad:#cf222e; --acc:#0b3d6b; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0d1117; --fg:#e6edf3; --mut:#9da7b3; --line:#30363d;
  --head:#161b22; --ok:#3fb950; --bad:#f85149; --acc:#79c0ff; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
  margin:0 auto; max-width:1200px; padding:16px; }}
h1 {{ color:var(--acc); margin-bottom:4px; }} h2 {{ color:var(--acc); border-bottom:2px solid var(--line);
  padding-bottom:4px; margin-top:40px; }} h3 {{ margin:20px 0 6px; font-size:16px; }} h4 {{ margin:14px 0 4px; }}
.what {{ color:var(--mut); }} code {{ background:var(--head); padding:0 3px; border-radius:3px; font-size:13px; }}
.tw {{ overflow-x:auto; margin:8px 0 14px; }} table {{ border-collapse:collapse; font-size:13.5px; min-width:50%; }}
th,td {{ border:1px solid var(--line); padding:4px 8px; text-align:left; vertical-align:top; }}
th {{ background:var(--head); }} td.ok {{ color:var(--ok); font-weight:600; }} td.bad {{ color:var(--bad); font-weight:600; }}
.figs {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(min(100%,520px),1fr)); gap:14px; }}
figure {{ margin:0; border:1px solid var(--line); border-radius:6px; padding:6px; background:#fff; }}
figure img {{ width:100%; height:auto; display:block; }} figcaption {{ font-size:12px; color:#555; text-align:center; }}
details {{ margin:10px 0; }} summary {{ cursor:pointer; color:var(--acc); }}
nav ul {{ columns:2; padding-left:18px; }} @media (max-width:600px) {{ nav ul {{ columns:1; }} }}
</style></head><body>
<h1>MedLift-3D results</h1>
<p class="what">Generated {datetime.datetime.now():%Y-%m-%d %H:%M} from <code>{html.escape(od)}</code></p>
<nav><ul><li><a href="#overview">Overview</a></li>{''.join(toc)}</ul></nav>
<div id="overview"></div>{''.join(parts)}
</body></html>"""
    with open(out, "w") as f:
        f.write(page)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB): {len(toc)} sections, "
          f"{sum(1 for v in ver.values() if v)} steps with checks, {len(fails)} failed checks")


if __name__ == "__main__":
    main()
