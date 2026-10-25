#!/usr/bin/env python
"""Step 12 - interactive click-to-3D demo (Gradio).

1. pick a case (or a synthetic phantom), scroll the DTS depth slider; cases are labelled with their
   patient split -- use *test* patients (never seen in training) for an honest demonstration;
   yellow circles are the detector's candidate nodules near that depth,
   green circles the radiologists' annotated nodules (LIDC ground truth)
2. click on a nodule in the coronal DTS slice
3. MedLift-3D reconstructs the nodule in 3D and reports volume and mass with
   intervals (calibrated if step 10 has been run), shows the 3D surface coloured
   by uncertainty, and can simulate a follow-up scan (VDT vs MDT)

    pip install gradio plotly
    python scripts/12_demo_app.py --share          # public link for your defense
"""
from _common import base_parser, setup

import numpy as np

from medlift3d.demo import DemoSession


def build_ui(sess: DemoSession):
    import gradio as gr
    import plotly.graph_objects as go

    from medlift3d import viz

    state = {"iy": 0}

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

    def on_case(name):
        if name.startswith("phantom:"):                     # on-the-fly synthetic chest (saved cases load from disk)
            sess.load_phantom(seed=int(name.split(":")[-1]))
        else:
            sess.load_case(name)
        ann = sess.annotations()
        iy = sess.n_depth // 2
        if ann:
            iy = ann[0]["iy"]
        elif sess.candidates:
            iy = sess.candidates[0]["index_zyx"][1]
        state["iy"] = iy
        cands = [[k + 1, *[round(v, 1) for v in c["xyz"]], round(c["score"], 1)] for k, c in enumerate(sess.candidates)]
        anns = [[a["index"] + 1, *[round(v, 1) for v in a["xyz"]], round(a["diameter_mm"], 1), a["iy"]] for a in ann]
        return (gr.Slider(minimum=0, maximum=sess.n_depth - 1, value=iy, step=1, label="DTS depth slice (y)"),
                sess.coronal_image(iy), cands, anns,
                f"Loaded {name}{split_note(name)}: {len(sess.candidates)} detector candidates (yellow), {len(ann)} "
                "radiologist-annotated nodules (green). Click a nodule.")

    def on_slider(iy):
        state["iy"] = int(iy)
        return sess.coronal_image(int(iy)), f"depth y = {sess.depth_mm(int(iy)):.0f} mm"

    def on_click(evt: gr.SelectData):
        col, row = evt.index
        xyz = sess.pixel_to_world(int(row), int(col), state["iy"])
        rep = sess.analyze(xyz)
        res = sess.last["res"]
        meth = rep["method"]
        e = res["methods"][meth]
        vols = {"DTS ROI-SIRT (input)": res["roi_sirt_hu"], f"{meth} (mean)": e["mean_volume"]}
        masks = {f"{meth} (mean)": e["vote_mask"]}
        stds = {f"{meth} (mean)": e["std_volume"]} if e.get("std_volume") is not None else None
        import os
        import time
        # a new file per click: the browser may still be loading the previous one
        path = viz.save_roi_panel(os.path.join(sess.cfg["paths"]["outputs"], "12_demo", f"roi_{time.time_ns()}.png"), vols, res["roi_vg"], xyz, masks, stds,
                                  "coronal = what DTS shows, sagittal = depth (where DTS is blind)")
        md = (f"### Nodule at ({xyz[0]:.0f}, {xyz[1]:.0f}, {xyz[2]:.0f}) mm\n"
              f"- **Mass:** {rep['mass_mg']:.0f} mg  [{rep['mass_interval'][0]:.0f} - {rep['mass_interval'][1]:.0f}]  ({rep['interval_type']})\n"
              f"- **Volume:** {rep['volume_mm3']:.0f} mm3  [{rep['volume_interval'][0]:.0f} - {rep['volume_interval'][1]:.0f}]\n"
              f"- 2D diameter on the DTS slice: {rep['d2d_long_mm']:.1f} mm\n"
              f"- EUPS volume category: {rep['eups_category']}\n"
              f"- method: {rep['method']}, {rep['n_found']} of {rep['n_members']} hypotheses found a nodule, {rep['seconds']:.1f} s\n"
              + (f"- **note:** {rep['note']}\n" if rep["note"] else "") + "\n"
              "_Mass is measured directly by every X-ray view; volume needs the depth extent, which DTS cannot see -- "
              "hence its wider interval._")
        return md, path, fig3d()

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
    choices = [(f"{c}  ({label.get(sess.case_split(c), 'split unknown')})", c) for c in sess.available_cases()]
    choices += [("phantom:0  (synthetic)", "phantom:0"), ("phantom:1  (synthetic)", "phantom:1")]
    with gr.Blocks(title="MedLift-3D") as demo:
        gr.Markdown("# MedLift-3D: click-to-3D lung nodule mass & growth from low-dose DTS")
        with gr.Row():
            with gr.Column(scale=1):
                case = gr.Dropdown(choices, label="case (test patients first: the model never saw them)", value=None)
                status = gr.Markdown()
                depth = gr.Slider(0, 100, step=1, label="DTS depth slice (y)")
                img = gr.Image(label="coronal DTS slice (click a nodule)", interactive=False)
                cands = gr.Dataframe(headers=["#", "x", "y", "z", "score"], label="detector candidates (yellow)")
                anns = gr.Dataframe(headers=["#", "x", "y", "z", "d (mm)", "depth slice"],
                                    label="radiologist-annotated nodules (green; LIDC ground truth, for the demo)")
            with gr.Column(scale=1):
                report = gr.Markdown()
                plot3d = gr.Plot(label="3D nodule: reconstruction (green-red) vs radiologists' nodule (grey)")
                panel = gr.Image(label="reconstruction (coronal / sagittal)")
                with gr.Row():
                    days = gr.Number(180, label="follow-up interval (days)")
                    dt = gr.Number(300, label="true doubling time (days, 0 = stable)")
                fu_btn = gr.Button("simulate follow-up scan")
                fu = gr.Markdown()
        case.change(on_case, [case], [depth, img, cands, anns, status])
        depth.change(on_slider, [depth], [img, status])
        img.select(on_click, None, [report, panel, plot3d])
        fu_btn.click(on_followup, [days, dt], [fu])
    demo.medlift_handlers = {"on_case": on_case, "on_slider": on_slider, "on_click": on_click,
                             "on_followup": on_followup, "state": state}
    return demo


def selftest(sess: DemoSession, out_dir: str) -> bool:
    """Headless check of the demo backend on the first available case."""
    import os

    from medlift3d.utils import Checker
    from medlift3d import viz

    ck = Checker("Step 12: demo backend self-test")
    cases = sess.available_cases()
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
    m = sess.mesh()
    ck.check("3D surface mesh", m is not None and len(m[1]) > 0, f"{0 if m is None else len(m[1])} faces")
    t = sess.truth_mesh()
    if m is not None and t is not None:                  # truth surface must sit where the reconstruction is
        gap = float(np.linalg.norm(m[0].mean(0) - t[0].mean(0)))
        ck.check("3D truth surface overlaps the reconstruction (centre distance)", gap < 10.0, f"{gap:.1f} mm", "< 10 mm")
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
        ck.check("UI: select case -> slice + candidates", out[1].ndim == 3, f"{len(out[2])} candidates")
        green = (out[1][..., 1] > 200) & (out[1][..., 0] < 60)
        ck.check("UI: annotated nodule marked in green at its depth", len(out[3]) == 0 or bool(green.any()),
                 f"{len(out[3])} annotated nodules")
        iy = int(round(sess.ctx.coarse_vg.world_to_index(np.asarray(nod.center_xyz))[1]))
        h["on_slider"](iy)
        vg = sess.ctx.coarse_vg
        ix, _, iz = np.round(vg.world_to_index(np.asarray(nod.center_xyz))).astype(int)
        row = (vg.shape[0] - 1) - iz
        s = sess.display_scale                                             # click in the enlarged image
        md, path, fig = h["on_click"](SimpleNamespace(index=[int(ix) * s + s // 2, int(row) * s + s // 2]))
        ck.check("UI: click on the nodule -> report + panel + 3D plot", "Mass" in md and os.path.exists(path), path)
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
    ap.add_argument("--members", type=int, default=4,
                    help="diffusion hypotheses per click (4: ~2x faster than the evaluation's 8)")
    args = ap.parse_args()
    cfg, dev = setup(args)
    sess = DemoSession(cfg, dev, n_members=args.members)
    print(f"method: {sess.method} | calibration: {'yes' if sess.calib else 'no'} | cases: {len(sess.available_cases())}")
    if args.selftest:
        import os

        ok = selftest(sess, os.path.join(cfg["paths"]["outputs"], "12_demo"))
        raise SystemExit(0 if ok else 1)
    build_ui(sess).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
