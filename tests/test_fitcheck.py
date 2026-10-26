"""Training-log parsing and the under/over-fitting verdicts (scripts/15_check_fit.py)."""
from medlift3d.fitcheck import detector_verdict, diffusion_verdict, parse_detector_log, parse_diffusion_log


def diff_log(gains, train_fit=None, heldout_fit=None, start=0, every=2000):
    lines = []
    for i, g in enumerate(gains):
        step = start + (i + 1) * every
        for s in range(step - every + 50, step + 1, 50):
            lines.append(f"step {s}/1000000  loss {0.1 / (1 + s / 5000):.4f}  lr 2.00e-04  0.700s/step  time left 3.00 h")
        lines.append(f"  [val] PSNR sample vs CT 20.00 dB | condition vs CT 18.00 dB  || nodule box: sample "
                     f"{11.0 + g:.2f} dB | condition 11.00 dB")
        if train_fit is not None:
            lines.append(f"  [fit] denoising loss at fixed noise (EMA model): train pairs {train_fit[i]:.4f} | "
                         f"held-out pairs {heldout_fit[i]:.4f}  (held-out / train 1.00)")
    return "\n".join(lines)


def det_log(cpms):
    lines = []
    for i, c in enumerate(cpms):
        step = (i + 1) * 500
        for s in range(step - 400, step + 1, 100):
            lines.append(f"step {s}/6000  loss {0.5 / (1 + s / 1000):.4f}  (1.0 min)")
        lines.append(f"[val] step {step}: CPM {c:.3f}, sensitivity at 2 FP/scan 0.30, recall top-10 0.40")
    return "\n".join(lines)


def test_parse_diffusion_log():
    d = parse_diffusion_log(diff_log([1.0, 2.0, 3.0], [0.05, 0.04, 0.03], [0.06, 0.05, 0.045]))
    assert [c["step"] for c in d["checks"]] == [2000, 4000, 6000]
    assert abs(d["checks"][1]["gain"] - 2.0) < 1e-6
    assert abs(d["checks"][2]["fit_heldout"] - 0.045) < 1e-6
    assert d["steps"][-1] == 6000


def test_parse_keeps_last_run_and_resumed_continuation():
    old = diff_log([1.0, 1.0, 1.0])
    new = diff_log([0.5, 1.5])
    more = diff_log([2.5], start=4000)                       # MORE_TRAIN_HOURS: continues the step count
    d = parse_diffusion_log(old + "\n" + new + "\n" + more)
    assert [c["step"] for c in d["checks"]] == [2000, 4000, 6000]
    assert abs(d["checks"][0]["gain"] - 0.5) < 1e-6


def test_diffusion_verdicts():
    assert diffusion_verdict(parse_diffusion_log(diff_log([1.0, 2.0])))["verdict"] == "TOO EARLY"
    up = diffusion_verdict(parse_diffusion_log(diff_log([0.5, 1.0, 1.5, 2.0, 2.6],
                                                        [.05, .045, .04, .035, .03], [.06, .055, .05, .046, .042])))
    assert up["verdict"].startswith("STILL IMPROVING")
    over = diffusion_verdict(parse_diffusion_log(diff_log([1.0, 2.5, 2.6, 2.4, 2.3, 2.2],
                                                          [.05, .04, .035, .03, .025, .02],
                                                          [.06, .05, .049, .051, .053, .056])))
    assert over["verdict"] == "OVERFITTING" and over["heldout_min_step"] == 6000
    flat = diffusion_verdict(parse_diffusion_log(diff_log([2.0, 2.4, 2.5, 2.45, 2.5, 2.48],
                                                          [.04, .038, .037, .037, .036, .036],
                                                          [.05, .049, .048, .048, .048, .048])))
    assert flat["verdict"].startswith("CONVERGED")
    dead = diffusion_verdict(parse_diffusion_log(diff_log([-1.0, -0.5, 0.0, -0.2, 0.1])))
    assert dead["verdict"] == "NOT LEARNING"
    gain_only = diffusion_verdict(parse_diffusion_log(diff_log([1.0, 3.0, 2.5, 2.0])))   # older log: no [fit]
    assert gain_only["verdict"] == "OVERFITTING"


def test_detector_verdicts():
    d = parse_detector_log(det_log([0.10, 0.155, 0.13, 0.12, 0.11]))      # like the 40-patient Kaggle run
    assert [c["step"] for c in d["checks"]] == [500, 1000, 1500, 2000, 2500]
    v = detector_verdict(d)
    assert v["verdict"] == "OVERFITTING" and v["best_step"] == 1000
    assert detector_verdict(parse_detector_log(det_log([0.1, 0.15, 0.2, 0.25])))["verdict"].startswith("STILL")
    assert detector_verdict(parse_detector_log(det_log([0.2, 0.24, 0.25, 0.25, 0.245])))["verdict"].startswith("CONV")
    assert detector_verdict(parse_detector_log(det_log([0.01, 0.02, 0.01])))["verdict"] == "NOT LEARNING"
