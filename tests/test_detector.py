"""Learned nodule detector (step 14): targets, peaks, FROC, and that the network learns."""
import numpy as np
import torch

from medlift3d.detector import (DetNet, focal_loss, froc, heatmap_peaks, heatmap_target, is_hit, predict_heatmap,
                                score_threshold)
from medlift3d.geometry import VolumeGeometry


def test_heatmap_target_peak_and_depth_elongation():
    vg = VolumeGeometry.centered((40, 40, 40), (2.5, 2.5, 2.5))
    t = heatmap_target(vg, [(5.0, -3.0, 7.0)], [8.0], min_sigma_mm=2.5, depth_factor=2.0)
    ix, iy, iz = np.round(vg.world_to_index(np.array([5.0, -3.0, 7.0]))).astype(int)
    assert t[iz, iy, ix] == 1.0 and t.max() == 1.0
    assert (t >= 0.999).sum() == 1                                   # exactly one positive voxel
    assert t[iz, iy + 2, ix] > t[iz, iy, ix + 2]                     # longer in depth (y) than in-plane


def test_peaks_nms_and_order():
    vg = VolumeGeometry.centered((40, 40, 40), (2.5, 2.5, 2.5))
    h = heatmap_target(vg, [(-20.0, 0.0, 0.0), (20.0, 0.0, 0.0)], [6.0, 6.0])
    h[:, :, 20:] *= 0.5                                              # second blob weaker
    c = heatmap_peaks(h, vg, top_k=5, min_score=0.1)
    assert len(c) == 2 and c[0]["score"] > c[1]["score"]
    assert abs(c[0]["xyz"][0] + 20.0) < 2.6 and abs(c[1]["xyz"][0] - 20.0) < 2.6


def test_hit_froc_and_threshold():
    nod = {"xyz": [0.0, 0.0, 0.0], "diameter_mm": 8.0}
    assert is_hit([5.0, 10.0, 0.0], nod["xyz"], 8.0)                # 5 mm in-plane, 10 mm deep: found
    assert not is_hit([9.0, 0.0, 0.0], nod["xyz"], 8.0)             # 9 mm in-plane: missed
    assert not is_hit([0.0, 20.0, 0.0], nod["xyz"], 8.0)            # 20 mm deep: missed
    tp = {"xyz": [0.0, 0.0, 0.0], "score": 0.9}
    fp = lambda s: {"xyz": [100.0, 0.0, 0.0], "score": s}            # noqa: E731
    scans = [{"cands": [tp, fp(0.8), {"xyz": [1.0, 0.0, 0.0], "score": 0.7}], "nodules": [nod]},   # dup: ignored
             {"cands": [fp(0.95), fp(0.5)], "nodules": [nod]}]                                       # missed
    r = froc(scans)
    assert r["n_nodules"] == 2 and r["recall_top10"] == 0.5
    assert r["sensitivity"][1.0] == 0.5 and r["sensitivity"][0.125] == 0.0   # the TP comes after 1 FP (0.5 per scan)
    assert abs(score_threshold(scans, fp_per_scan=1.0) - 0.5) < 1e-9          # 2 scans x 1 FP: the 3rd FP's score


def test_detector_learns_bright_blobs():
    """A tiny DetNet trained for a few hundred CPU steps finds blobs in a new volume."""
    rng = np.random.default_rng(0)
    vg = VolumeGeometry.centered((32, 32, 32), (2.5, 2.5, 2.5))
    xs, ys, zs = vg.axes()
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")

    def volume(centres):
        v = -850.0 + 40.0 * rng.standard_normal(vg.shape)
        for c in centres:
            v += 800.0 * np.exp(-((X - c[0]) ** 2 + (Z - c[2]) ** 2) / (2 * 3.0 ** 2) - (Y - c[1]) ** 2 / (2 * 8.0 ** 2))
        return v.astype(np.float32)

    def random_centres(n):
        return [tuple(rng.uniform(-25, 25, 3)) for _ in range(n)]

    torch.manual_seed(0)
    model = DetNet(8)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(150):
        cs = random_centres(2)
        x = torch.as_tensor((volume(cs) + 300.0) / 700.0)[None, None]
        t = torch.as_tensor(heatmap_target(vg, cs, [6.0, 6.0]))[None, None]
        loss = focal_loss(model(x), t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    cs = [(-15.0, 5.0, 10.0), (15.0, -5.0, -12.0)]
    c = heatmap_peaks(predict_heatmap(model, volume(cs)), vg, top_k=2)
    found = sum(any(is_hit(k["xyz"], ci, 6.0) for k in c) for ci in cs)
    assert found == 2, c
