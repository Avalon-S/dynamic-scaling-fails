"""What share of decode memory the recurrent state accounts for.

In a hybrid the attention KV cache grows with context while the recurrent state does not,
so the state's share falls as context grows. Computed from the published configurations;
no GPU required.
"""
MODELS = {
    # Qwen3.5-0.8B, from the HF config: 24 layers on a full_attention_interval of 4,
    # so 18 linear and 6 full; GQA with 2 KV heads of dim 256; the recurrent state is
    # (num_value_heads, key_head_dim, value_head_dim) held in fp32.
    "Qwen3.5-0.8B": dict(
        linear_layers=18, attn_layers=6, kv_heads=2, head_dim=256, kv_bytes=2,
        state_elems=16 * 128 * 128, state_bytes=4),
    # Mamba-2-1.3B is a pure SSM: no attention layers, so no KV cache at all.
    "Mamba-2-1.3B": dict(
        linear_layers=48, attn_layers=0, kv_heads=0, head_dim=0, kv_bytes=2,
        state_elems=64 * 64 * 128, state_bytes=4),
}


def report(name, m, lengths=(2048, 8192, 32768, 131072)):
    state_mb = m["linear_layers"] * m["state_elems"] * m["state_bytes"] / 2**20
    kv_per_tok = 2 * m["attn_layers"] * m["kv_heads"] * m["head_dim"] * m["kv_bytes"]
    print("%s: recurrent state %.1f MB (fixed), KV %.0f bytes/token"
          % (name, state_mb, kv_per_tok))
    print("   %10s %12s %12s %10s" % ("context", "KV cache", "state", "state share"))
    for L in lengths:
        kv_mb = kv_per_tok * L / 2**20
        share = state_mb / (state_mb + kv_mb) if (state_mb + kv_mb) else 1.0
        print("   %10d %9.1f MB %9.1f MB %9.1f%%" % (L, kv_mb, state_mb, 100 * share))
    print()


if __name__ == "__main__":
    for k, v in MODELS.items():
        report(k, v)
    print("The state is the whole cache in a pure SSM, and a minority of it in this")
    print("hybrid once the context passes a few thousand tokens.")
