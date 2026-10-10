"""Loading Qwen3.5, reaching its recurrent state, and fake-quantizing it.

Qwen3.5 is a Gated DeltaNet hybrid with a 262K context. At 0.8B, 18 of its 24 layers are
linear-attention layers, and each keeps an fp32 state of shape (1, 16, 128, 128). The
state lives at

  cache.layers[i].recurrent_states[0]      (transformers 5.x DynamicCache)

Built against transformers 5.14.1 and torch 2.7.1.
"""
import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

QWEN35_08B = "Qwen/Qwen3.5-0.8B"
QWEN35_2B = "Qwen/Qwen3.5-2B"


def load(model_id=QWEN35_08B, dtype=torch.bfloat16, device="cuda"):
    # bfloat16, which is the config's own dtype, and not float16. Under INT4 the state
    # compounds to a norm of 1e5 and beyond, which overflows float16's +-65504 exponent
    # range to NaN and truncates the trajectory partway through a long run. bfloat16 has
    # the same exponent range as fp32, so the divergence stays measurable. The state is
    # stored in fp32 either way; the overflow is in the compute dtype.
    tok = AutoTokenizer.from_pretrained(model_id)
    model = Qwen3_5ForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return tok, model


def get_rstates(cache):
    """[(layer_idx, tensor)] for the linear-attention layers only. The tensors are the
    live cache entries, so writing to them in place is what quantizes the state."""
    out = []
    for i, lyr in enumerate(cache.layers):
        rs = getattr(lyr, "recurrent_states", None)
        if rs is not None and len(rs) > 0 and rs[0] is not None:
            out.append((i, rs[0]))
    return out


_ABS_SCALE = {}   # layer index -> a fixed scale, calibrated once and then held


def fake_quant(x, bits, dim=-1, stochastic=False, abs_scale=None, cap=None, pct=None, clip=True):
    """symmetric per-channel absmax fake-quant.

    stochastic: round to nearest, or stochastic rounding, which is unbiased.
    abs_scale:  a fixed scale that does not follow the state's amplitude. This is the
                persistent-static policy.
    cap:        a per-channel ceiling, so that s = min(absmax/qmax, cap). Resolution
                stays dynamic while the state is well behaved, and once drift pushes
                absmax/qmax above the ceiling the scale stops following the state, which
                is what cuts the feedback. A ceiling of zero is the static policy and an
                infinite one is plain dynamic. The ceiling is calibrated on healthy
                states, so on a healthy state it never binds.
    """
    if bits is None:
        return x
    qmax = 2 ** (bits - 1) - 1
    if abs_scale is not None:
        s = abs_scale
    elif pct is not None:                 # percentile baseline: a quantile, not the max
        flat = x.abs()
        s = torch.quantile(flat.float(), pct, dim=dim, keepdim=True).clamp_min(1e-8) / qmax
        s = s.to(x.dtype)
    else:
        s = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-8) / qmax
        if cap is not None:
            s = torch.minimum(s, cap)
    y = x / s
    if stochastic:
        q = torch.floor(y + torch.rand_like(y))
    else:
        q = torch.round(y)
    if clip:                              # clip=False leaves the codes unclamped
        q = torch.clamp(q, -qmax - 1, qmax)
    return (q * s).to(x.dtype)


def quantize_states(cache, bits, dim=-1, stochastic=False):
    """Fake-quantize the linear-attention states in place.

    bits is None to leave them alone, an int for a uniform width, 'zero' to discard
    the state, or a list or dict to set a width per layer."""
    if bits is None or cache is None:
        return 0
    rs = get_rstates(cache)
    n = 0
    for idx, (i, t) in enumerate(rs):
        b = (bits[idx] if idx < len(bits) else None) if isinstance(bits, (list, tuple)) \
            else bits.get(i) if isinstance(bits, dict) else bits
        if b is None:
            continue
        q = torch.zeros_like(t) if b == "zero" \
            else fake_quant(t.float(), b, dim, stochastic).to(t.dtype)
        t.copy_(q)
        n += 1
    return n
