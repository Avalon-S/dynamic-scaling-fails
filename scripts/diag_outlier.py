"""Step-by-step trace of the quantizer scale and the state norm.

Runs dynamic INT4 and static INT4 closed-loop on the same text and records the mean scale
and the state norm at every step, which is what makes the compounding visible as a curve
rather than as an endpoint.
"""
import json
import os
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from qwen35_common import load, get_rstates          # noqa: E402
from data import get_filler_ids                      # noqa: E402
from qwen35_range import calib_absmax                # noqa: E402

torch.set_grad_enabled(False)
tok, model = load()
H = 2048
cfull = get_filler_ids(tok, 8000, part="calib")
calib_docs = [cfull[i * 2200:i * 2200 + 2100] for i in range(3)]
absmax = calib_absmax(model, calib_docs)          # the persistent scale, fixed here
test = get_filler_ids(tok, H + 500, part="test")


def measure(h):
    """h is the pre-quantization state, (1,16,128,128).
    Returns (scale_mean, outlier_ratio, bulk_levels, norm)."""
    a = h.float().abs()
    scale = (a.amax(-1, keepdim=True) / 7).clamp_min(1e-12)   # per-channel dynamic scale
    flat = a.flatten()
    p999 = torch.quantile(flat, 0.999).item()
    p50 = torch.quantile(flat, 0.5).item()
    return (scale.mean().item(), p999 / max(p50, 1e-9),
            p50 / max(scale.mean().item(), 1e-12), h.float().norm().item())


def run(mode):
    cache = None
    tel = {"scale": [], "outlier": [], "levels": [], "norm": [], "clip": []}
    for k in range(H):
        out = model(torch.tensor([[test[k]]], device=model.device),
                    past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        for i, t in get_rstates(cache):
            sm, orr, lv, nm = measure(t)
            if mode == "dynamic":
                s = (t.float().abs().amax(-1, keepdim=True) / 7).clamp_min(1e-12)
                clip = 0.0
            else:  # static
                s = (absmax[i].to(t.device) / 7).clamp_min(1e-12)
                raw = torch.round(t.float() / s)
                clip = ((raw < -8) | (raw > 7)).float().mean().item()
            code = torch.clamp(torch.round(t.float() / s), -8, 7)
            t.copy_((code * s).to(t.dtype))
            # telemetry from layer 0 only; the others track it and cost storage
            if i == get_rstates(cache)[0][0]:
                tel["scale"].append(sm); tel["outlier"].append(orr)
                tel["levels"].append(lv); tel["norm"].append(nm); tel["clip"].append(clip)
    return tel


res = {}
for mode in ("dynamic", "static"):
    res[mode] = run(mode)
    t = res[mode]
    # three points in the decode: early, middle, late
    for lbl, j in [("t=10", 10), ("t=256", 256), ("t=1024", 1024), ("t=2047", 2047)]:
        print(f"  {mode:8s} {lbl:8s} scale={t['scale'][j]:.3e} outlier={t['outlier'][j]:.1f} "
              f"levels={t['levels'][j]:.2f} norm={t['norm'][j]:.2e} clip={t['clip'][j]:.4f}", flush=True)

json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "results",
                                 "outlier_diag_08b.json"), "w"))
print("[saved] outlier_diag_08b.json", flush=True)
