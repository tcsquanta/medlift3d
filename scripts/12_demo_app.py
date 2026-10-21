#!/usr/bin/env python
"""Step 12 - interactive click-to-3D demo (Gradio).

1. pick a case (or a synthetic phantom), scroll the DTS depth slider;
   yellow circles are the detector's candidate nodules near that depth
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
        m = sess.mesh()
        if m is None:
            return go.Figure()
        xyz, faces, col = m
        f = go.Figure(go.Mesh3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                intensity=col, colorscale="RdYlGn_r", colorbar=dict(title="std (HU)"), opacity=1.0))
        f.update_layout(scene=dict(xaxis_title="x (mm)", yaxis_title="depth y (mm)", zaxis_title="z (mm)",
                                   aspectmode="data"), margin=dict(l=0, r=0, t=30, b=0),
                        title="nodule surface: green = X-rays agree, red = uncertain (depth)")
        return f

    def on_case(name):
        if name.startswith("phantom"):
            sess.load_phantom(seed=int(name.split(":")[-1]) if ":" in name else 0)
        else:
            sess.load_case(name)
        iy = sess.n_depth // 2
        if sess.candidates:
            iy = sess.candidates[0]["index_zyx"][1]
        state["iy"] = iy
        cands = [[k + 1, *[round(v, 1) for v in c["xyz"]], round(c["score"], 1)] for k, c in enumerate(sess.candidates)]
        return (gr.update(maximum=sess.n_depth - 1, value=iy), sess.coronal_image(iy), cands,
                f"Loaded {name}: {len(sess.candidates)} candidates. Click a nodule.")

    def on_slider(iy):
        state["iy"] = int(iy)
        return sess.coronal_image(int(iy)), f"depth y = {sess.depth_mm(int(iy)):.0f} mm"

    def on_click(evt: gr.SelectData):
        col, row = evt.index
        xyz = sess.pixel_to_world(int(row), int(col), state["iy"])
        rep = sess.analyze(xyz)
        res = sess.last["res"]
        e = res["methods"][sess.method]
        vols = {"DTS ROI-SIRT (input)": res["roi_sirt_hu"], f"{sess.method} (mean)": e["mean_volume"]}
        masks = {f"{sess.method} (mean)": e["vote_mask"]}
        stds = {f"{sess.method} (mean)": e["std_volume"]} if e.get("std_volume") is not None else None
        import os
        path = viz.save_roi_panel(os.path.join(sess.cfg["paths"]["outputs"], "12_demo", "last_roi.png"), vols, res["roi_vg"], xyz, masks, stds,
                                  "coronal = what DTS shows, sagittal = depth (where DTS is blind)")
        md = (f"### Nodule at ({xyz[0]:.0f}, {xyz[1]:.0f}, {xyz[2]:.0f}) mm\n"
              f"- **Mass:** {rep['mass_mg']:.0f} mg  [{rep['mass_interval'][0]:.0f} - {rep['mass_interval'][1]:.0f}]  ({rep['interval_type']})\n"
              f"- **Volume:** {rep['volume_mm3']:.0f} mm3  [{rep['volume_interval'][0]:.0f} - {rep['volume_interval'][1]:.0f}]\n"
              f"- 2D diameter on the DTS slice: {rep['d2d_long_mm']:.1f} mm\n"
              f"- EUPS volume category: {rep['eups_category']}\n"
              f"- method: {rep['method']}, {rep['n_members']} hypotheses, {rep['seconds']:.1f} s\n\n"
              "_Mass is measured directly by every X-ray view; volume needs the depth extent, which DTS cannot see -- "
              "hence its wider interval._")
        return md, path, fig3d()

    def on_followup(days, dt):
        try:
            g = sess.followup(float(days), float(dt) if float(dt) > 0 else 1e9)
        except Exception as e:
            return f"Follow-up not possible: {e}"
        return (f"### Virtual follow-up after {g['interval_days']:.0f} days\n"
                f"- true doubling time: {g['true_doubling_days']:.0f} d -> **{g['true_category']}**\n"
                f"- volume doubling time (VDT): {g['vdt_days']:.0f} d -> {g['vdt_category']}\n"
                f"- mass doubling time (MDT): {g['mdt_days']:.0f} d -> {g['mdt_category']}")

    choices = sess.available_cases() + ["phantom:0", "phantom:1"]
    with gr.Blocks(title="MedLift-3D") as demo:
        gr.Markdown("# MedLift-3D: click-to-3D lung nodule mass & growth from low-dose DTS")
        with gr.Row():
            with gr.Column(scale=1):
                case = gr.Dropdown(choices, label="case", value=None)
                status = gr.Markdown()
                depth = gr.Slider(0, 100, step=1, label="DTS depth slice (y)")
                img = gr.Image(label="coronal DTS slice (click a nodule)", interactive=False)
                cands = gr.Dataframe(headers=["#", "x", "y", "z", "score"], label="candidate nodules")
            with gr.Column(scale=1):
                report = gr.Markdown()
                plot3d = gr.Plot(label="3D nodule (uncertainty colour)")
                panel = gr.Image(label="reconstruction (coronal / sagittal)")
                with gr.Row():
                    days = gr.Number(180, label="follow-up interval (days)")
                    dt = gr.Number(300, label="true doubling time (days, 0 = stable)")
                fu_btn = gr.Button("simulate follow-up scan")
                fu = gr.Markdown()
        case.change(on_case, [case], [depth, img, cands, status])
        depth.change(on_slider, [depth], [img, status])
        img.select(on_click, None, [report, panel, plot3d])
        fu_btn.click(on_followup, [days, dt], [fu])
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
    ck.check("case loads, candidates computed", sess.ctx is not None, f"{cases[0]}: {len(sess.candidates)} candidates")
    img = sess.coronal_image(sess.n_depth // 2)
    ck.check("coronal slice image", img.ndim == 3 and img.shape[2] == 3, str(img.shape))
    nod = sess.case.nodules[0]
    rep = sess.analyze(nod.center_xyz)
    ck.check("analysis report finite", np.isfinite(rep["volume_mm3"]) and np.isfinite(rep["mass_mg"]),
             f"V {rep['volume_mm3']:.0f} mm3 [{rep['volume_interval'][0]:.0f}-{rep['volume_interval'][1]:.0f}], "
             f"M {rep['mass_mg']:.0f} mg [{rep['mass_interval'][0]:.0f}-{rep['mass_interval'][1]:.0f}] ({rep['interval_type']})")
    m = sess.mesh()
    ck.check("3D surface mesh", m is not None and len(m[1]) > 0, f"{0 if m is None else len(m[1])} faces")
    try:
        g = sess.followup(180.0, 300.0)
        ck.check("virtual follow-up -> VDT/MDT", np.isfinite(g["mdt_days"]) or np.isinf(g["mdt_days"]),
                 f"true {g['true_doubling_days']:.0f} d | VDT {g['vdt_days']:.0f} d | MDT {g['mdt_days']:.0f} d")
    except Exception as e:
        ck.check("virtual follow-up -> VDT/MDT", False, str(e))
    try:
        import gradio  # noqa: F401

        build_ui(sess)
        ck.check("Gradio UI builds", True, gradio.__version__)
    except ImportError:
        ck.info("Gradio UI", "gradio not installed (pip install gradio plotly) -- backend verified only")
    os.makedirs(out_dir, exist_ok=True)
    res = sess.last["res"]
    e = res["methods"][sess.method]
    viz.save_roi_panel(os.path.join(out_dir, "selftest_roi.png"), {"input": res["roi_sirt_hu"], sess.method: e["mean_volume"]},
                       res["roi_vg"], nod.center_xyz, {sess.method: e["vote_mask"]}, None, "demo self-test")
    return ck.finish(out_dir)


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--selftest", action="store_true", help="test the backend headless and exit")
    args = ap.parse_args()
    cfg, dev = setup(args)
    sess = DemoSession(cfg, dev)
    print(f"method: {sess.method} | calibration: {'yes' if sess.calib else 'no'} | cases: {len(sess.available_cases())}")
    if args.selftest:
        import os

        ok = selftest(sess, os.path.join(cfg["paths"]["outputs"], "12_demo"))
        raise SystemExit(0 if ok else 1)
    build_ui(sess).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
