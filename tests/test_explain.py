"""Seen / unseen split, saliency and correlation helpers (medlift3d.explain, step 17)."""
import numpy as np
import torch

from medlift3d.explain import cgls_split, depth_profile, saliency_share_within, skill, spearman_ci, split_metrics


def test_split_is_exact_orthogonal_and_null():
    torch.manual_seed(0)
    A = torch.randn(30, 80)                                 # 30 measurements, 80 unknowns: big null space
    fwd, adj = (lambda v: A @ v), (lambda r: A.T @ r)
    x = torch.randn(80)
    seen, unseen, info = cgls_split(x, fwd, adj, n_iter=200, tol=1e-10)
    assert torch.allclose(seen + unseen, x, atol=1e-5)
    assert info["rel_residual"] < 1e-4                      # the unseen part changes no measurement
    assert abs(info["cos_seen_unseen"]) < 1e-3              # orthogonal pieces
    P = torch.linalg.pinv(A) @ A                            # exact projector onto the row space
    assert torch.allclose(seen, P @ x, atol=1e-4)


def test_split_of_seen_volume_has_no_unseen_part():
    torch.manual_seed(1)
    A = torch.randn(20, 50)
    x = A.T @ torch.randn(20)                               # lies in the row space: fully measured
    seen, unseen, _ = cgls_split(x, lambda v: A @ v, lambda r: A.T @ r, n_iter=200, tol=1e-10)
    assert float(unseen.norm()) < 1e-3 * float(x.norm())


def test_split_on_projector_box():
    """Real projector, 3 views of a small box: a blob elongated in depth is mostly unseen."""
    from medlift3d.geometry import VolumeGeometry
    from medlift3d.projector import Projector

    vg = VolumeGeometry.centered((16, 24, 16), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    P = Projector(vg, step_mm=0.5)
    zs, xs = np.meshgrid(np.linspace(-7, 7, 15), np.linspace(-7, 7, 15), indexing="ij")
    o, d = [], []
    for ang in np.deg2rad([-15, 0, 15]):                    # parallel rays along y, tilted in x
        for z, xx in zip(zs.ravel(), xs.ravel()):
            o.append([xx - 30 * np.sin(ang), -30 * np.cos(ang), z])
            d.append([np.sin(ang), np.cos(ang), 0.0])
    o, d = torch.tensor(o, dtype=torch.float32), torch.tensor(d, dtype=torch.float32)
    zz, yy, xx = np.meshgrid(*[np.arange(n) - (n - 1) / 2 for n in vg.shape], indexing="ij")
    blob = np.exp(-(xx ** 2 + zz ** 2) / 8 - yy ** 2 / 8).astype(np.float32)
    x = torch.as_tensor(blob)
    seen, unseen, info = cgls_split(x, lambda v: P(v, o, d), lambda r: P.backproject(r, o, d), n_iter=100)
    assert info["rel_residual"] < 0.02
    m = split_metrics(blob, seen.numpy(), unseen.numpy())
    assert 0.2 < m["unseen_energy_share"] < 0.95             # depth shape largely invisible to 3 views
    assert abs(m["unseen_mass_share"]) < 0.15                # ... but the mass is seen (H1)
    prof = depth_profile(seen.numpy(), (8, 12, 8))
    assert prof.argmax() in range(9, 15)


def test_metrics_helpers():
    ref = np.array([1.0, -1.0, 2.0])
    assert skill(ref, ref) == 1.0 and abs(skill(np.zeros(3), ref)) < 1e-12 and skill(-ref, ref) < 0
    sal = np.zeros((9, 9, 9))
    sal[4, 4, 4] = 3.0
    sal[0, 0, 0] = 1.0
    assert abs(saliency_share_within(sal, (4, 4, 4), 2.0, (1, 1, 1)) - 0.75) < 1e-9
    rng = np.random.default_rng(0)
    a = rng.normal(size=60)
    rho, lo, hi, n = spearman_ci(a, a + 0.3 * rng.normal(size=60), n_boot=300)
    assert n == 60 and lo < rho < hi and rho > 0.8
