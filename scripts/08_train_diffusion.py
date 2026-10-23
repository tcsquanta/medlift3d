#!/usr/bin/env python
"""Step 8 - train the conditional 3D diffusion prior (the GPU-heavy step).

Run on a rented GPU (Vast.ai RTX 4090 / A100 recommended):

    python scripts/08_train_diffusion.py --steps 40000 --batch-size 8

Check the code first on any machine (CPU is fine, ~2-5 min):

    python scripts/08_train_diffusion.py --tiny

Every --sample-every steps it samples a few validation ROIs with the EMA
model and logs PSNR(sample, CT) next to PSNR(condition, CT); the run is
healthy when the first number rises above the second. Checkpoints:
work/models/diffusion/{last.pt, best.pt}; log: train_log.csv; figures in
outputs/08_diffusion/.

Several GPUs (e.g. Kaggle "GPU T4 x2"): launch with torchrun; each GPU gets
--batch-size samples per step (DistributedDataParallel), validation and
checkpoints run on the first GPU:

    torchrun --standalone --nproc_per_node 2 scripts/08_train_diffusion.py --batch-size 4 --amp fp16
"""
import csv
import datetime
import math
import os
import sys
import time

from _common import base_parser, setup

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from medlift3d.diffusion import EMA, GaussianDiffusion, unit_to_hu
from medlift3d.diffusion.checkpoint import build_model, save_checkpoint
from medlift3d.diffusion.dataset import RoiPairDataset
from medlift3d.diffusion.unet3d import count_parameters
from medlift3d.io import Paths
from medlift3d.metrics import psnr
from medlift3d.utils import Checker
from medlift3d import viz


def infinite(loader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)                     # new shuffle every pass, same on every GPU
        for b in loader:
            yield b
        epoch += 1


def tiny_pairs(n=6, size=32, seed=0):
    """Synthetic (blurred-in-y, noisy) -> sharp blob pairs for the --tiny check."""
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        z, y, x = np.mgrid[:size, :size, :size].astype(np.float32) - size / 2 + rng.uniform(-3, 3, 3)[:, None, None, None]
        r = rng.uniform(3, 7)
        tgt = -850 + 870 / (1 + np.exp((np.sqrt(x ** 2 + y ** 2 + z ** 2) - r) / 0.8))
        cond = -850 + ndimage.gaussian_filter1d(tgt + 850, 4.0, axis=1) + rng.normal(0, 20, tgt.shape)
        out.append((cond.astype(np.float32), tgt.astype(np.float32)))
    return out


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, pairs):
        from medlift3d.diffusion import hu_to_unit
        self.items = [(torch.from_numpy(hu_to_unit(c))[None], torch.from_numpy(hu_to_unit(t))[None]) for c, t in pairs]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _box(arr, cidx, half_vox):
    if cidx is None or np.any(np.asarray(cidx) < 0):
        return arr
    lo = [max(0, int(round(c - h))) for c, h in zip(cidx, half_vox)]
    hi = [min(n, int(round(c + h)) + 1) for c, n, h in zip(cidx, arr.shape, half_vox)]
    return arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]


@torch.no_grad()
def evaluate(model, diffusion, val_items, steps, device, gen_seed=0, start_frac=1.0, box_half_vox=None):
    """Mean PSNR of samples and of the condition vs CT: whole prism and nodule box."""
    model.eval()
    ps_s, ps_c, bx_s, bx_c, vis = [], [], [], [], []
    g = torch.Generator(device=device).manual_seed(gen_seed)
    for item in val_items:
        cond, tgt = item[0], item[1]
        cidx = item[2] if len(item) > 2 else None
        c = cond[None].to(device)
        x = diffusion.ddim_sample(model, c, steps=steps, eta=0.0, generator=g, start_frac=start_frac)
        s_hu = np.clip(unit_to_hu(x[0, 0]).float().cpu().numpy(), -1000, 400)
        c_hu = np.clip(unit_to_hu(cond[0]).numpy(), -1000, 400)
        t_hu = np.clip(unit_to_hu(tgt[0]).numpy(), -1000, 400)
        ps_s.append(psnr(s_hu, t_hu, 1400.0))
        ps_c.append(psnr(c_hu, t_hu, 1400.0))
        if box_half_vox is not None:
            bx_s.append(psnr(_box(s_hu, cidx, box_half_vox), _box(t_hu, cidx, box_half_vox), 1400.0))
            bx_c.append(psnr(_box(c_hu, cidx, box_half_vox), _box(t_hu, cidx, box_half_vox), 1400.0))
        vis.append((c_hu, s_hu, t_hu))
    model.train()
    box = (float(np.mean(bx_s)), float(np.mean(bx_c))) if bx_s else (float("nan"), float("nan"))
    return float(np.mean(ps_s)), float(np.mean(ps_c)), vis, box


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--amp", choices=["auto", "bf16", "fp16", "none"], default="auto")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--sample-every", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop after this many hours of training (evaluates + saves first); the learning rate "
                         "is constant after warm-up, so stopping early loses nothing")
    ap.add_argument("--tiny", action="store_true", help="CPU-sized sanity run on synthetic pairs")
    args = ap.parse_args()
    cfg, dev = setup(args)
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    if world > 1:                                        # launched by torchrun: one process per GPU
        local = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            dev = torch.device("cuda", local)
        else:
            dev = torch.device("cpu")
        dist.init_process_group("nccl" if dev.type == "cuda" else "gloo", timeout=datetime.timedelta(minutes=60))
        torch.manual_seed(int(cfg["seed"]) + rank)      # different diffusion noise per GPU (weights: rank 0's)
        np.random.seed(int(cfg["seed"]) + rank)
        if rank != 0:
            sys.stdout = open(os.devnull, "w")           # one log, from the first GPU
    main_proc = rank == 0
    paths = Paths(cfg)
    dcfg = dict(cfg["diffusion"])
    out = paths.out("08_diffusion" + ("_tiny" if args.tiny else ""))
    mdir = os.path.join(cfg["paths"]["work"], "models", "diffusion" + ("_tiny" if args.tiny else ""))
    os.makedirs(mdir, exist_ok=True)
    ck = Checker("Step 8: diffusion training" + (" (tiny sanity run)" if args.tiny else ""))

    if args.tiny:
        dcfg.update(base_ch=8, ch_mults=[1, 2, 2], num_res=1, attn_levels=[2], dropout=0.0, timesteps=200)
        steps = args.steps or 600
        bs = args.batch_size or 4
        train_ds = TinyDataset(tiny_pairs(6, 32, seed=0))
        val_items = [train_ds[i] for i in range(2)]
        args.sample_every = min(args.sample_every, steps)
        sample_steps = 25
    else:
        steps = args.steps or int(dcfg["train_steps"])
        bs = args.batch_size or int(dcfg["batch_size"])
        train_ds = RoiPairDataset(os.path.join(cfg["paths"]["work"], "roi_pairs", "train"), augment=True)
        vdir = os.path.join(cfg["paths"]["work"], "roi_pairs", "val")
        val_ds = RoiPairDataset(vdir, augment=False) if os.path.isdir(vdir) and os.listdir(vdir) else None
        src = RoiPairDataset(val_ds.files if val_ds is not None else train_ds.files, augment=False, return_meta=True)
        idx = np.linspace(0, len(src) - 1, min(args.n_val, len(src))).astype(int)
        val_items = [src[i] for i in idx]
        sample_steps = int(dcfg["sample_steps"])
    lr = args.lr or float(dcfg["lr"])
    model = build_model(dcfg).to(dev)
    diffusion = GaussianDiffusion(int(dcfg["timesteps"]), dcfg.get("prediction", "v"))
    ema = EMA(model, float(dcfg["ema_decay"]) if not args.tiny else 0.99)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    start = 0
    last_path = os.path.join(mdir, "last.pt")
    if args.resume and os.path.exists(last_path):
        st = torch.load(last_path, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        ema.shadow.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        start = int(st["step"])
        ema.n_updates = start
        print(f"resumed from step {start}")
    ck.info("parameters", f"{count_parameters(model) / 1e6:.2f} M")
    ck.info("training pairs", len(train_ds))
    use_cuda = dev.type == "cuda"
    if args.amp == "auto":
        amp = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if use_cuda else "none"
    else:
        amp = args.amp
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16" and use_cuda))
    except (AttributeError, TypeError):                      # torch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=(amp == "fp16" and use_cuda))
    ck.info("device / precision / batch", f"{dev} / {amp} / {bs} per GPU x {world} GPU(s)")
    net = model
    if world > 1:
        net = DistributedDataParallel(model, device_ids=[dev.index] if dev.type == "cuda" else None)
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    workers = 0 if args.tiny else max(1, min(args.workers, (os.cpu_count() or 2) // world))
    loader = DataLoader(train_ds, batch_size=bs, shuffle=sampler is None, sampler=sampler,
                        drop_last=len(train_ds) >= bs * world, num_workers=workers, pin_memory=use_cuda,
                        persistent_workers=False)
    it = infinite(loader, sampler)
    log_path = os.path.join(mdir, "train_log.csv")
    new_log = not (args.resume and os.path.exists(log_path))
    logf = open(log_path if main_proc else os.devnull, "w" if new_log else "a", newline="")
    logw = csv.writer(logf)
    if new_log:
        logw.writerow(["step", "loss", "lr", "sec_per_step", "val_psnr_sample", "val_psnr_condition"])
    warmup = float(max(1, min(500, steps // 10)))
    losses, hist_s, hist_c, hist_bs, hist_bc = [], [], [], [], []
    vsp = np.asarray(cfg["roi"]["voxel_mm"], float)[::-1]                              # (z, y, x)
    box_half = None if args.tiny else (np.asarray(cfg["roi"].get("box_mm", (40, 50, 40)), float)[::-1] / vsp / 2.0)
    best = -1e9
    t_start = t_last = time.perf_counter()
    model.train()
    for step in range(start + 1, steps + 1):
        cond, tgt = next(it)
        cond = cond.to(dev, non_blocking=True)
        tgt = tgt.to(dev, non_blocking=True)
        cur_lr = lr * min(1.0, step / warmup)
        for gparam in opt.param_groups:
            gparam["lr"] = cur_lr
        with torch.autocast(device_type=dev.type, dtype=dtype, enabled=dtype is not None):
            loss = diffusion.loss(net, tgt, cond)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        ema.update(model)
        losses.append(loss.item())
        if step % args.log_every == 0 or step == steps:
            dt = (time.perf_counter() - t_last) / args.log_every
            t_last = time.perf_counter()
            ml = float(np.mean(losses[-args.log_every:]))
            eta_h = dt * (steps - step) / 3600
            if args.max_hours is not None:                       # the time budget ends the run, not --steps
                eta_h = max(0.0, args.max_hours - (time.perf_counter() - t_start) / 3600)
            print(f"step {step}/{steps}  loss {ml:.4f}  lr {cur_lr:.2e}  {dt:.3f}s/step  "
                  f"{'time left' if args.max_hours is not None else 'ETA'} {eta_h:.2f} h")
            logw.writerow([step, ml, cur_lr, dt, "", ""])
            logf.flush()
        out_of_time = args.max_hours is not None and time.perf_counter() - t_start >= 3600.0 * args.max_hours
        if world > 1 and args.max_hours is not None:     # all GPUs must stop at the same step
            flag = torch.tensor([1.0 if out_of_time else 0.0], device=dev)
            dist.broadcast(flag, 0)
            out_of_time = bool(flag.item())
        if out_of_time:
            print(f"time budget of {args.max_hours:g} h reached at step {step}: evaluating, saving, stopping")
        if main_proc and (step % args.sample_every == 0 or step == steps or out_of_time):
            ps, pc, vis, (bxs, bxc) = evaluate(ema.shadow, diffusion, val_items, sample_steps, dev,
                                               start_frac=float(dcfg.get("start_frac", 1.0)), box_half_vox=box_half)
            hist_s.append(ps)
            hist_c.append(pc)
            hist_bs.append(bxs)
            hist_bc.append(bxc)
            logw.writerow([step, "", "", "", ps, pc])
            logf.flush()
            print(f"  [val] PSNR sample vs CT {ps:.2f} dB | condition vs CT {pc:.2f} dB"
                  + (f"  || nodule box: sample {bxs:.2f} dB | condition {bxc:.2f} dB" if np.isfinite(bxs) else ""))
            c_hu, s_hu, t_hu = vis[0]
            from medlift3d.geometry import VolumeGeometry

            n = c_hu.shape[0]
            vg = VolumeGeometry.centered(c_hu.shape, tuple(cfg["roi"]["voxel_mm"]) if not args.tiny else (1, 1, 1))
            viz.save_roi_panel(os.path.join(out, f"val_step{step:06d}.png"),
                               {"condition": c_hu, "diffusion sample": s_hu, "CT": t_hu}, vg, vg.center,
                               title=f"step {step}: PSNR sample {ps:.1f} dB / condition {pc:.1f} dB")
            save_checkpoint(last_path, model, ema, opt, step, dcfg)
            score = bxs - bxc if np.isfinite(bxs) else ps - pc        # gain over the condition, where it matters
            if score > best:
                best = score
                save_checkpoint(os.path.join(mdir, "best.pt"), model, ema, opt, step, dcfg)
        if out_of_time:
            break
    logf.close()
    if world > 1:
        dist.barrier()                                   # wait for the first GPU's last validation
        dist.destroy_process_group()
        if not main_proc:
            return
    viz.save_curve(os.path.join(out, "loss.png"), {"loss": np.convolve(losses, np.ones(min(50, len(losses))) / min(50, len(losses)), mode="valid")},
                   "diffusion training loss (moving average)", "step", "MSE(eps)", logy=True)
    k = max(1, len(losses) // 10)
    first, last = float(np.mean(losses[:k])), float(np.mean(losses[-k:]))
    ck.check("training loss decreased (last 10% / first 10%)", last < 0.7 * first, f"{first:.4f} -> {last:.4f}", "< 0.7")
    if hist_s:
        if hist_bs and np.isfinite(hist_bs[-1]):
            ck.check("diffusion samples beat the physics-only condition in the nodule box (PSNR)",
                     hist_bs[-1] > hist_bc[-1], f"{hist_bs[-1]:.2f} dB vs {hist_bc[-1]:.2f} dB")
            ck.info("   whole prism (incl. chest walls) PSNR sample / condition", f"{hist_s[-1]:.2f} / {hist_c[-1]:.2f} dB")
        else:
            ck.check("diffusion samples beat the physics-only condition on validation ROIs (PSNR)",
                     hist_s[-1] > hist_c[-1], f"{hist_s[-1]:.2f} dB vs {hist_c[-1]:.2f} dB")
    ck.info("checkpoints", mdir)
    ck.finish(out)


if __name__ == "__main__":
    main()
