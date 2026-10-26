#!/usr/bin/env python
"""Step 12 - interactive click-to-3D demo (Gradio).

1. pick a new patient's DTS scan: by default only *test* patients (never seen by the diffusion
   model or the detector in training); --splits test cal train offers the others too, labelled
2. the nodule detector (step 14; training-free LoG if not trained) suggests possible nodules:
   yellow rings on the coronal DTS slice + a table with their probabilities
3. pick a suggestion -> "Reconstruct in 3D" (or click anywhere on the slice): MedLift-3D
   reconstructs it with the diffusion prior + Gaussians and reports mass and volume with
   intervals (calibrated if step 10 has been run), the 3D surface coloured by uncertainty,
   and can simulate a follow-up scan (VDT vs MDT)
4. tick "show radiologists' ground truth": green rings, which suggestions were real nodules,
   and the radiologists' mass / volume next to MedLift-3D's (LIDC consensus)

    pip install gradio plotly
    python scripts/12_demo_app.py --selftest       # headless check, must end ALL PASS
    python scripts/12_demo_app.py --share          # public link for your defense
"""
from _common import base_parser, setup

import numpy as np

from medlift3d.demo import DemoSession


def build_ui(sess: DemoSession, splits=None):
    import gradio as gr
    import plotly.graph_objects as go

    from medlift3d import viz

    state = {"iy": 0, "truth": False, "selected": None, "name": "", "gt": None}

    def fig3d():
        m, t = sess.mesh(), sess.truth_mesh()
        f = go.Figure()
        if m is not None:
            xyz, faces, col = m
            f.add_trace(go.Mesh3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                  intensity=col, colorscale="RdYlGn_r", colorbar=dict(title="std (HU)"), opacity=1.0,
                                  name="MedLift-3D reconstruction", showlegend=True))
        if t is not None:                                   # radiologists' nodule (ground truth), see-through
            txyz, tfaces = t
            f.add_trace(go.Mesh3d(x=txyz[:, 0], y=txyz[:, 1], z=txyz[:, 2], i=tfaces[:, 0], j=tfaces[:, 1],
                                  k=tfaces[:, 2], color="lightgrey", opacity=0.35, name="radiologists' nodule (truth)",
                                  showlegend=True))
        if m is None and t is None:
            return f.update_layout(title="no nodule surface (nothing segmented at the click)")
        title = ("green = hypotheses agree, red = uncertain (depth); grey = radiologists' nodule"
                 if m is not None else "nothing segmented at the click; grey = radiologists' nodule")
        f.update_layout(scene=dict(xaxis_title="x (mm)", yaxis_title="depth y (mm)", zaxis_title="z (mm)",
                                   aspectmode="data"), margin=dict(l=0, r=0, t=40, b=0), title=title,
                        legend=dict(orientation="h", y=-0.02),
                        annotations=[dict(text="drag to rotate, scroll to zoom", showarrow=False, x=0, y=1.0,
                                          xref="paper", yref="paper", font=dict(size=11))])
        return f

    def split_note(name):
        sp = "phantom" if name.startswith("phantom:") else sess.case_split(name)
        return {"test": " (test patient: never seen in training)",
                "cal": " (calibration patient: not used for training)",
                "train": " (**training patient**: the model has seen this patient; pick a test case for an honest demo)",
                "phantom": " (synthetic phantom)"}.get(sp, "")

    def image():
        return sess.coronal_image(state["iy"], with_annotations=state["truth"], selected=state["selected"])

    def cand_rows():
        rows = []
        for k, c in enumerate(sess.candidates):
            m = c.get("match")
            truth = ("-" if not state["truth"] else ("no (false suggestion)" if m is None else f"yes: nodule {m + 1}"))
            rows.append([k + 1, round(c["score"], 3), *[round(v, 1) for v in c["xyz"]], c["index_zyx"][1], truth])
        return rows

    def ann_rows():
        if not state["truth"]:
            return []
        summ = sess.detection_summary()
        return [[a["index"] + 1, *[round(v, 1) for v in a["xyz"]], round(a["diameter_mm"], 1), a["iy"],
                 f"suggestion #{summ['rank_of_nodule'][a['index']]}" if a["index"] in summ["rank_of_nodule"] else "missed"]
                for a in sess.annotations()]

    def status_text():
        n = len(sess.candidates)
        txt = (f"Loaded {state['name']}{split_note(state['name'])}. The {sess.detector_name} suggests **{n} possible "
               "nodules** (yellow rings; pick one below or click anywhere on the slice).")
        if state["truth"]:
            d = sess.detection_summary()
            txt += (f"\n\n**Compared with the radiologists (green):** {len(d['found'])} of {d['n_nodules']} annotated "
                    f"nodules found among the {d['n_suggestions']} suggestions; {d['n_false']} suggestions are not "
                    "annotated nodules (LIDC lists only nodules >= 4 mm marked by >= 3 of 4 radiologists).")
        return txt

    def pick_choices():
        return [f"#{k + 1}  (p = {c['score']:.2f}, depth slice {c['index_zyx'][1]})" for k, c in enumerate(sess.candidates)]

    def report_md():
        if sess.last is None:
            return ""
        rep, xyz = sess.last["report"], sess.last["click"]

        def num(v):
            return f"{v:.0f}" if v >= 10 else (f"{v:.1f}" if v >= 0.1 else "<0.1")

        hyp = ", ".join(f"{num(m['mass_mg'])} mg / {num(m['volume_mm3'])} mm3" if m["reason"] == "found"
                        else "nothing" for m in rep.get("members", []))
        flag = ("\n> **Low confidence:** the hypotheses disagree strongly or most found nothing -- the depth "
                "information at this point is too weak for a reliable measurement.\n" if rep.get("low_confidence") else "")
        md = (f"### Nodule at ({xyz[0]:.0f}, {xyz[1]:.0f}, {xyz[2]:.0f}) mm\n" + flag +
              f"- **Mass:** {num(rep['mass_mg'])} mg  [{num(rep['mass_interval'][0])} - {num(rep['mass_interval'][1])}]  ({rep['interval_type']})\n"
              f"- **Volume:** {num(rep['volume_mm3'])} mm3  [{num(rep['volume_interval'][0])} - {num(rep['volume_interval'][1])}]\n"
              f"- 2D diameter on the DTS slice: {rep['d2d_long_mm']:.1f} mm\n"
              f"- EUPS volume category: {rep['eups_category']}\n"
              + (f"- **EUPS category, calibrated probability** (step 16): " + " | ".join(
                  f"{c}: {100 * p:.0f}%" for c, p in rep["category_probs"].items()) + "\n"
                 if rep.get("category_probs") else "")
              + f"- method: {rep['method']}, {rep['n_found']} of {rep['n_members']} hypotheses found a nodule, {rep['seconds']:.1f} s\n"
              + (f"- the hypotheses (mass / volume): {hyp}\n" if hyp else "")
              + (f"- **note:** {rep['note']}\n" if rep["note"] else "") + "\n"
              "_Mass is measured directly by every X-ray view; volume needs the depth extent, which DTS cannot see -- "
              "hence its wider interval._")
        if state["truth"]:
            g = state.get("gt")
            if g is None:
                md += "\n\n### Ground truth\nNo radiologist-annotated nodule within 15 mm of this point."
            else:
                def err(est, true):
                    return f"{100 * (est / true - 1):+.0f}%" if true > 0 and est > 1e-2 else "not measurable"

                md += (f"\n\n### Ground truth (LIDC radiologists, nodule {g['index'] + 1})\n"
                       f"| | MedLift-3D | radiologists | error |\n|---|---|---|---|\n"
                       f"| mass (mg) | {rep['mass_mg']:.0f} | {g['mass_mg']:.0f} | {err(rep['mass_mg'], g['mass_mg'])} |\n"
                       f"| volume (mm3) | {rep['volume_mm3']:.0f} | {g['volume_mm3']:.0f} | {err(rep['volume_mm3'], g['volume_mm3'])} |\n"
                       f"| diameter (mm) | {rep['d2d_long_mm']:.1f} | {g['diameter_mm']:.1f} | |\n\n"
                       f"Mass inside the interval: **{'yes' if rep['mass_interval'][0] <= g['mass_mg'] <= rep['mass_interval'][1] else 'no'}**; "
                       f"volume inside the interval: **{'yes' if rep['volume_interval'][0] <= g['volume_mm3'] <= rep['volume_interval'][1] else 'no'}**.")
        return md

    def run_analysis(xyz):
        import os
        import time

        sess.analyze(xyz)
        state["gt"] = sess.ground_truth_at(xyz)
        res = sess.last["res"]
        meth = sess.last["method"]
        e = res["methods"][meth]
        vols = {"DTS ROI-SIRT (input)": res["roi_sirt_hu"], f"{meth} (mean)": e["mean_volume"]}
        masks = {f"{meth} (mean)": e["vote_mask"]}
        stds = {f"{meth} (mean)": e["std_volume"]} if e.get("std_volume") is not None else None
        # a new file per click: the browser may still be loading the previous one
        path = viz.save_roi_panel(os.path.join(sess.cfg["paths"]["outputs"], "12_demo", f"roi_{time.time_ns()}.png"),
                                  vols, res["roi_vg"], xyz, masks, stds,
                                  "coronal = what DTS shows, sagittal = depth (where DTS is blind)")
        return report_md(), path, fig3d()

    def on_case(name):
        if name.startswith("phantom:"):                     # on-the-fly synthetic chest (saved cases load from disk)
            sess.load_phantom(seed=int(name.split(":")[-1]))
        else:
            sess.load_case(name)
        state.update(name=name, selected=None, gt=None)
        iy = sess.candidates[0]["index_zyx"][1] if sess.candidates else sess.n_depth // 2
        state["iy"] = iy
        return (gr.Slider(minimum=0, maximum=sess.n_depth - 1, value=iy, step=1, label="DTS depth slice (y)"),
                image(), cand_rows(), ann_rows(), status_text(),
                gr.Dropdown(choices=pick_choices(), value=None, label="AI suggestion to reconstruct"))

    def on_slider(iy):
        state["iy"] = int(iy)
        return image(), f"depth y = {sess.depth_mm(int(iy)):.0f} mm"

    def on_truth(show):
        state["truth"] = bool(show)
        if sess.ctx is None:
            return None, [], [], "", ""
        return image(), cand_rows(), ann_rows(), status_text(), report_md()

    def on_pick(choice):
        if not choice or sess.ctx is None:
            return gr.Slider(), None, "Pick a suggestion first.", None, None
        k = int(choice.split()[0].lstrip("#")) - 1
        c = sess.candidates[k]
        state["selected"], state["iy"] = k, int(c["index_zyx"][1])
        md, path, fig = run_analysis(c["xyz"])
        return gr.Slider(value=state["iy"]), image(), md, path, fig

    def on_click(evt: gr.SelectData):
        col, row = evt.index
        state["selected"] = None
        return run_analysis(sess.pixel_to_world(int(row), int(col), state["iy"]))

    def on_followup(days, dt):
        try:
            g = sess.followup(float(days), float(dt) if float(dt) > 0 else 1e9)
        except Exception as e:
            return f"Follow-up not possible: {e}"
        def days(d):
            return "no growth" if d == float("inf") else ("not measurable" if d != d else f"{d:.0f} d")

        return (f"### Virtual follow-up after {g['interval_days']:.0f} days\n"
                f"- true doubling time: {days(g['true_doubling_days'])} -> **{g['true_category']}**\n"
                f"- volume doubling time (VDT): {days(g['vdt_days'])} -> {g['vdt_category']}\n"
                f"- mass doubling time (MDT): {days(g['mdt_days'])} -> {g['mdt_category']}")

    label = {"test": "test, unseen", "cal": "calibration", "train": "training patient"}
    choices = [(f"{c}  ({label.get(sess.case_split(c), 'split unknown')})", c) for c in sess.available_cases(splits)]
    choices += [("phantom:0  (synthetic)", "phantom:0"), ("phantom:1  (synthetic)", "phantom:1")]
    with gr.Blocks(title="MedLift-3D") as demo:
        gr.Markdown("# MedLift-3D: click-to-3D lung nodule mass & growth from low-dose DTS\n"
                    "1. pick a new patient's DTS scan  2. the detector suggests possible nodules  "
                    "3. reconstruct a suggestion in 3D (the diffusion prior fills in the depth)  "
                    "4. tick *ground truth* to compare with the radiologists")
        with gr.Row():
            with gr.Column(scale=1):
                case = gr.Dropdown(choices, label="patient DTS scan (test patients: never seen in training)", value=None)
                status = gr.Markdown()
                depth = gr.Slider(0, 100, step=1, label="DTS depth slice (y)")
                img = gr.Image(label="coronal DTS slice (yellow = AI suggestion, cyan = selected; click anywhere)",
                               interactive=False)
                with gr.Row():
                    pick = gr.Dropdown([], label="AI suggestion to reconstruct")
                    pick_btn = gr.Button("Reconstruct in 3D", variant="primary")
                cands = gr.Dataframe(headers=["#", "probability", "x", "y", "z", "depth slice", "radiologists' nodule?"],
                                     label="AI-suggested nodules (yellow)")
                truth = gr.Checkbox(False, label="show radiologists' ground truth (green) -- for evaluation")
                anns = gr.Dataframe(headers=["#", "x", "y", "z", "d (mm)", "depth slice", "found by the AI?"],
                                    label="radiologist-annotated nodules (LIDC ground truth)")
            with gr.Column(scale=1):
                report = gr.Markdown()
                plot3d = gr.Plot(label="3D nodule: reconstruction (green-red) vs radiologists' nodule (grey)")
                panel = gr.Image(label="reconstruction (coronal / sagittal)")
                with gr.Row():
                    days = gr.Number(180, label="follow-up interval (days)")
                    dt = gr.Number(300, label="true doubling time (days, 0 = stable)")
                fu_btn = gr.Button("simulate follow-up scan")
                fu = gr.Markdown()
        case.change(on_case, [case], [depth, img, cands, anns, status, pick])
        depth.change(on_slider, [depth], [img, status])
        truth.change(on_truth, [truth], [img, cands, anns, status, report])
        pick_btn.click(on_pick, [pick], [depth, img, report, panel, plot3d])
        img.select(on_click, None, [report, panel, plot3d])
        fu_btn.click(on_followup, [days, dt], [fu])
    demo.medlift_handlers = {"on_case": on_case, "on_slider": on_slider, "on_click": on_click, "on_pick": on_pick,
                             "report_md": report_md,
                             "on_truth": on_truth, "on_followup": on_followup, "state": state}
    return demo


def selftest(sess: DemoSession, out_dir: str) -> bool:
    """Headless check of the demo backend on the first available case."""
    import os

    from medlift3d.utils import Checker
    from medlift3d import viz

    ck = Checker("Step 12: demo backend self-test")
    cases = sess.available_cases(["test"]) or sess.available_cases()
    if not cases:
        ck.check("cases available", False, "run steps 4-6 first")
        return ck.finish(out_dir)
    sess.load_case(cases[0])
    ck.check("case loads, candidates computed", sess.ctx is not None, f"{cases[0]} ({sess.case_split(cases[0])} split): {len(sess.candidates)} candidates")
    img = sess.coronal_image(sess.n_depth // 2)
    ck.check("coronal slice image", img.ndim == 3 and img.shape[2] == 3, str(img.shape))
    nod = max(sess.case.nodules, key=lambda n: n.diameter_mm)          # the clearest nodule of the case
    ck.info("test nodule", f"d = {nod.diameter_mm:.1f} mm at {[round(v) for v in nod.center_xyz]}")
    rep = sess.analyze(nod.center_xyz)
    ck.check("analysis report finite", np.isfinite(rep["volume_mm3"]) and np.isfinite(rep["mass_mg"]),
             f"V {rep['volume_mm3']:.0f} mm3 [{rep['volume_interval'][0]:.0f}-{rep['volume_interval'][1]:.0f}], "
             f"M {rep['mass_mg']:.0f} mg [{rep['mass_interval'][0]:.0f}-{rep['mass_interval'][1]:.0f}] ({rep['interval_type']})")
    ck.info("method used", rep["method"] + (f" ({rep['note']})" if rep["note"] else ""))
    if sess.cps is not None:
        cp = rep.get("category_probs")
        ck.check("calibrated EUPS category probabilities (step 16) sum to 1" if cp else
                 "calibrated EUPS category probabilities (step 16)", cp is None or abs(sum(cp.values()) - 1) < 1e-6,
                 ", ".join(f"{k}: {100 * v:.0f}%" for k, v in cp.items()) if cp else
                 f"not shown (no step-16 calibration for {rep['method']}, or no nodule was measured)")
    m = sess.mesh()
    ck.check("3D surface mesh", m is not None and len(m[1]) > 0, f"{0 if m is None else len(m[1])} faces")
    t = sess.truth_mesh()
    ck.check("3D truth surface sits on the annotated nodule (centre distance; coordinate check)",
             t is not None and float(np.linalg.norm(t[0].mean(0) - np.asarray(nod.center_xyz))) < 0.5 * nod.diameter_mm + 2.0,
             "no surface" if t is None else f"{np.linalg.norm(t[0].mean(0) - np.asarray(nod.center_xyz)):.1f} mm")
    if m is not None and t is not None:                  # accuracy, not a code check: reported, not judged
        ck.info("reconstruction surface vs radiologists' surface (centre distance)",
                f"{np.linalg.norm(m[0].mean(0) - t[0].mean(0)):.1f} mm"
                + (f"; {rep['note']}" if rep["note"] else "") + (" [low confidence]" if rep.get("low_confidence") else ""))
    try:
        g = sess.followup(180.0, 300.0)
        ck.check("virtual follow-up -> VDT/MDT", np.isfinite(g["mdt_days"]) or np.isinf(g["mdt_days"]),
                 f"true {g['true_doubling_days']:.0f} d | VDT {g['vdt_days']:.0f} d | MDT {g['mdt_days']:.0f} d")
    except Exception as e:
        ck.check("virtual follow-up -> VDT/MDT", False, str(e))
    try:
        import gradio  # noqa: F401
        from types import SimpleNamespace

        ui = build_ui(sess)
        ck.check("Gradio UI builds", True, gradio.__version__)
        h = ui.medlift_handlers
        out = h["on_case"](cases[0])
        ck.check("UI: select case -> slice + AI suggestions", out[1].ndim == 3 and len(out[2]) == len(sess.candidates),
                 f"{len(out[2])} suggestions by the {sess.detector_name}; ground truth hidden: {len(out[3]) == 0}")
        iy = int(round(sess.ctx.coarse_vg.world_to_index(np.asarray(nod.center_xyz))[1]))
        h["on_slider"](iy)                                                 # to the nodule's depth
        img_t, _, ann_t, status_t, _ = h["on_truth"](True)
        green = (img_t[..., 1] > 200) & (img_t[..., 0] < 60)
        ck.check("UI: ground truth shown on request (green ring at the nodule's depth)", bool(green.any()) and len(ann_t) > 0,
                 f"{len(ann_t)} annotated nodules")
        d = sess.detection_summary()
        ck.info("AI suggestions vs radiologists on this case", f"{len(d['found'])} of {d['n_nodules']} nodules found, "
                f"{d['n_false']} of {d['n_suggestions']} suggestions not annotated")
        if sess.candidates:
            _, _, md_p, path_p, _ = h["on_pick"](f"#1  (p = {sess.candidates[0]['score']:.2f})")
            ck.check("UI: reconstruct AI suggestion #1 -> report (+ ground truth table) + panel",
                     "Mass" in md_p and "Ground truth" in md_p and os.path.exists(path_p), md_p.splitlines()[0])
            h["on_slider"](iy)                                             # picking moved the slider: back to the nodule
        vg = sess.ctx.coarse_vg
        ix, _, iz = np.round(vg.world_to_index(np.asarray(nod.center_xyz))).astype(int)
        row = (vg.shape[0] - 1) - iz
        s = sess.display_scale                                             # click in the enlarged image
        md, path, fig = h["on_click"](SimpleNamespace(index=[int(ix) * s + s // 2, int(row) * s + s // 2]))
        ck.check("UI: click on the nodule -> report + ground truth + panel + 3D plot",
                 "Mass" in md and "radiologists, nodule" in md and os.path.exists(path), path)
        fu_md = h["on_followup"](180, 300)
        ck.check("UI: follow-up button", "doubling" in fu_md, fu_md.splitlines()[0][:120])
    except ImportError:
        ck.info("Gradio UI", "gradio not installed (pip install gradio plotly) -- backend verified only")
    os.makedirs(out_dir, exist_ok=True)
    res, meth = sess.last["res"], sess.last["method"]
    e = res["methods"][meth]
    viz.save_roi_panel(os.path.join(out_dir, "selftest_roi.png"), {"input": res["roi_sirt_hu"], meth: e["mean_volume"]},
                       res["roi_vg"], nod.center_xyz, {meth: e["vote_mask"]}, None, "demo self-test")
    return ck.finish(out_dir)


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--selftest", action="store_true", help="test the backend headless and exit")
    ap.add_argument("--splits", nargs="+", default=["test"],
                    help="patients offered in the demo (default: test = never seen in training; e.g. --splits test cal train)")
    ap.add_argument("--members", type=int, default=4,
                    help="diffusion hypotheses per click (4: ~2x faster than the evaluation's 8)")
    args = ap.parse_args()
    cfg, dev = setup(args)
    sess = DemoSession(cfg, dev, n_members=args.members)
    print(f"method: {sess.method} | detector: {sess.detector_name} | calibration: {'yes' if sess.calib else 'no'} | "
          f"cases offered ({'/'.join(args.splits)}): {len(sess.available_cases(args.splits))} | "
          f"category probabilities (step 16): {'yes' if sess.cps else 'no'}")
    if args.selftest:
        import os

        ok = selftest(sess, os.path.join(cfg["paths"]["outputs"], "12_demo"))
        raise SystemExit(0 if ok else 1)
    build_ui(sess, args.splits).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
