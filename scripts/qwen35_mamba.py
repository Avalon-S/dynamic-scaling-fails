"""Cross-architecture control on Mamba-2-1.3B.

The SSM state is inference_params.key_value_memory_dict[layer][1], shape
(b, heads, head_dim, d_state). Every policy quantizes along d_state per
(head, head_dim), which is the axis corresponding to the one used for Gated DeltaNet.
Protocol matches the matched-target design: fixed context, only the last H steps
quantized, same continuation scored.
"""
import argparse, json, os, sys, statistics as st, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import get_filler_ids                                            # noqa: E402
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel             # noqa: E402
from mamba_ssm.utils.generation import InferenceParams                     # noqa: E402
from transformers import AutoTokenizer                                     # noqa: E402

DEV = "cuda"


def get_ssm(inf):
    return [(li, inf.key_value_memory_dict[li][1]) for li in sorted(inf.key_value_memory_dict)]


def q_dyn(inf, absmax, qmax):
    for li, s in get_ssm(inf):
        sc = (s.float().abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-12)
        s.copy_((torch.clamp(torch.round(s.float() / sc), -(qmax + 1), qmax) * sc).to(s.dtype))


def q_static(inf, absmax, qmax):
    for li, s in get_ssm(inf):
        sc = (absmax[li].to(s.device) / qmax).clamp_min(1e-12)
        s.copy_((torch.clamp(torch.round(s.float() / sc), -(qmax + 1), qmax) * sc).to(s.dtype))


def q_zero(inf, absmax, qmax):
    for li, s in get_ssm(inf):
        s.zero_()


def ssm_norm(inf):
    return st.mean([s.float().norm().item() for _, s in get_ssm(inf)])


@torch.no_grad()
def new_inf(maxlen):
    return InferenceParams(max_seqlen=maxlen, max_batch_size=1)


@torch.no_grad()
def calib_absmax(model, docs, maxlen):
    """Collect the absmax per (head, head_dim) along d_state from a healthy decode,
    over several sequences and several points in each."""
    SNAP = {128, 512, 1024, 1536, 2000}
    acc = {}
    for doc in docs:
        inf = new_inf(maxlen)
        for k in range(min(2048, len(doc))):
            model(torch.tensor([[doc[k]]], device=DEV), inference_params=inf)
            inf.seqlen_offset += 1
            if k in SNAP:
                for li, s in get_ssm(inf):
                    a = s.float().abs().amax(-1, keepdim=True)   # (1,64,64,1)
                    acc[li] = torch.maximum(acc[li], a) if li in acc else a
    return acc


@torch.no_grad()
def doc_nll(model, doc, CTX, H, cont, qfn, absmax):
    inf = new_inf(CTX + cont + 8)
    warm = CTX - H
    for k in range(warm):                              # unquant warm-up
        model(torch.tensor([[doc[k]]], device=DEV), inference_params=inf); inf.seqlen_offset += 1
    for k in range(warm, CTX):                         # the quantized exposure
        out = model(torch.tensor([[doc[k]]], device=DEV), inference_params=inf); inf.seqlen_offset += 1
        if qfn: qfn(inf, absmax, 7)
    nm = ssm_norm(inf)
    cur = out.logits[:, -1].float(); nll = 0.0
    for j in range(cont):                              # score the continuation, still quantizing
        gt = doc[CTX + j]
        nll -= torch.log_softmax(cur, -1)[0, gt].item()
        out = model(torch.tensor([[gt]], device=DEV), inference_params=inf); inf.seqlen_offset += 1
        if qfn: qfn(inf, absmax, 7)
        cur = out.logits[:, -1].float()
    return nll / cont, nm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Hs", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--CTX", type=int, default=2048)
    ap.add_argument("--cont", type=int, default=256)
    ap.add_argument("--n_docs", type=int, default=8)
    ap.add_argument("--stride", type=int, default=5000)
    ap.add_argument("--independent", action="store_true")
    ap.add_argument("--out", default="results/mamba_08b.json")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained("state-spaces/mamba2-1.3b", device=DEV,
                                             dtype=torch.bfloat16).eval()
    cfull = get_filler_ids(tok, 8000, part="calib")
    calib_docs = [cfull[i * 2200:i * 2200 + 2100] for i in range(3)]
    absmax = calib_absmax(model, calib_docs, a.CTX + a.cont + 8)
    if getattr(a, "independent", False):
        from data import get_articles
        docs = [art[:a.CTX + a.cont] for art in
                get_articles(tok, a.n_docs, a.CTX + a.cont, part="test")]
        print(f"[docs] {len(docs)} independent articles", flush=True)
    else:
        test = get_filler_ids(tok, a.n_docs * a.stride + a.CTX + a.cont + 100, part="test")
        docs = [test[i * a.stride:i * a.stride + a.CTX + a.cont] for i in range(a.n_docs)]
    conds = {"unquant": None, "dynamic4": q_dyn, "static4": q_static, "zero": q_zero}
    print(f"[calib] mamba2-1.3b done | Hs={a.Hs} n_docs={a.n_docs}", flush=True)

    res = {H: {c: dict(nll=[], norm=[]) for c in conds} for H in a.Hs}
    for di, doc in enumerate(docs):
        for H in a.Hs:
            for c, fn in conds.items():
                nll, nm = doc_nll(model, doc, a.CTX, H, a.cont, fn, absmax)
                res[H][c]["nll"].append(nll); res[H][c]["norm"].append(nm)
        print(f"  doc {di+1}/{a.n_docs} done", flush=True)

    summary = {}
    for H in a.Hs:
        summary[H] = {}
        uf = res[H]["unquant"]["nll"]
        for c in conds:
            exc = [res[H][c]["nll"][i] - uf[i] for i in range(a.n_docs)]
            summary[H][c] = dict(excess=st.mean(exc), norm=st.mean(res[H][c]["norm"]), exc=exc)
        print(f"  H={H}: " + " ".join(f"{c}={summary[H][c]['excess']:.3f}(norm{summary[H][c]['norm']:.1e})"
                                      for c in conds), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(summary=summary, config=vars(a)), open(a.out, "w"), indent=2)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
