"""Diffusion model wiring: shapes, noise schedule, a tiny overfit run, sampler."""
import numpy as np
import torch

from medlift3d.diffusion import GaussianDiffusion, UNet3D, hu_to_unit, unit_to_hu
from medlift3d.diffusion.unet3d import count_parameters


def _tiny_model():
    return UNet3D(in_ch=2, out_ch=1, base_ch=8, ch_mults=(1, 2), num_res=1, attn_levels=(1,), dropout=0.0)


def test_unet_shapes_and_default_size():
    m = _tiny_model()
    x = torch.randn(2, 2, 16, 16, 16)
    y = m(x, torch.tensor([3, 500]))
    assert y.shape == (2, 1, 16, 16, 16)
    big = UNet3D()                                       # default config used for training
    n = count_parameters(big)
    assert 5e6 < n < 3e7, n


def test_unit_scaling_roundtrip():
    hu = np.array([-1000.0, -300.0, 400.0])
    assert np.allclose(hu_to_unit(hu), [-1, 0, 1])
    assert np.allclose(unit_to_hu(hu_to_unit(hu)), hu)


def test_noise_schedule_endpoints():
    d = GaussianDiffusion(1000)
    assert float(d.ac[0]) > 0.999 and float(d.ac[-1]) < 1e-3
    x0 = torch.ones(512, 1, 2, 2, 2)
    xt = d.q_sample(x0, torch.full((512,), 999), torch.randn_like(x0))
    assert abs(float(xt.mean())) < 0.1 and abs(float(xt.std()) - 1.0) < 0.1


def test_tiny_overfit_loss_decreases():
    torch.manual_seed(0)
    m = _tiny_model()
    d = GaussianDiffusion(100)
    x0 = torch.tanh(torch.randn(2, 1, 8, 8, 8))
    cond = x0 + 0.3 * torch.randn_like(x0)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    losses = []
    for _ in range(150):
        opt.zero_grad()
        loss = d.loss(m, x0, cond)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert np.mean(losses[-20:]) < 0.6 * np.mean(losses[:20]), (np.mean(losses[:20]), np.mean(losses[-20:]))


def test_ddim_sampler_shape_determinism_and_dc_hook():
    m = _tiny_model().eval()
    d = GaussianDiffusion(100)
    cond = torch.zeros(2, 1, 8, 8, 8)
    g1 = torch.Generator().manual_seed(5)
    g2 = torch.Generator().manual_seed(5)
    a = d.ddim_sample(m, cond, steps=10, eta=0.0, generator=g1)
    b = d.ddim_sample(m, cond, steps=10, eta=0.0, generator=g2)
    assert a.shape == (2, 1, 8, 8, 8) and torch.allclose(a, b)
    calls = []

    def dc(x0):
        calls.append(1)
        return x0 * 0.0                                   # force the estimate to zero
    c = d.ddim_sample(m, cond, steps=10, eta=0.0, generator=torch.Generator().manual_seed(5), dc_fn=dc,
                      dc_start_frac=0.0)
    assert len(calls) == 10 and float(c.abs().max()) < 1e-5
