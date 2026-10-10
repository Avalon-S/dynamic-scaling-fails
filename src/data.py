"""The haystack corpus: WikiText-103, split so that nothing leaks."""


def get_filler_ids(tok, need_tokens, part="all"):
    """A stretch of plain English prose to use as the haystack.

    Downloads the one test-split parquet (about 1 MB) directly rather than through
    load_dataset(), which issues a HEAD against the 1.6 GB train parquet even when only
    the test split is asked for.

    part: 'all' (the default), 'calib' (the first half of the paragraphs), or 'test' (the
    second half). calib and test are cut at paragraph boundaries and do not overlap, so a
    scale calibrated on one is not calibrated on the filler the other is scored over.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id="Salesforce/wikitext", repo_type="dataset",
                        filename="wikitext-103-raw-v1/test-00000-of-00001.parquet")
    df = pd.read_parquet(p)
    paras = [t.strip() for t in df["text"] if len(t.strip()) >= 50 and not t.strip().startswith("=")]
    half = len(paras) // 2
    if part == "calib":
        paras = paras[:half]
    elif part == "test":
        paras = paras[half:]
    buf, n = [], 0
    for t in paras:
        buf.append(t); n += len(t)
        if n > need_tokens * 6:                # characters per token, generously
            break
    ids = tok("\n".join(buf), add_special_tokens=False).input_ids
    if len(ids) < need_tokens:
        raise RuntimeError(f"filler({part}) too short: {len(ids)} < {need_tokens}")
    return ids


def get_articles(tok, n_docs, min_tokens, part="test"):
    """n_docs genuinely independent WikiText articles, each at least min_tokens long.

    Cut at level-1 headings (' = Title = ') rather than at a fixed stride, so that two
    adjacent windows of one article are never counted as two samples.

    part: 'calib' takes the first half of the articles, 'test' the second half; the two do
    not share an article. Returns a list of token-id lists.
    """
    import re
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id="Salesforce/wikitext", repo_type="dataset",
                        filename="wikitext-103-raw-v1/test-00000-of-00001.parquet")
    df = pd.read_parquet(p)
    # level-1 headings only: ' = Title = ' has a single = each side, ' = = ' is a section
    title_re = re.compile(r"^ = [^=].* = $")
    articles, cur = [], []
    for t in df["text"]:
        s = t.rstrip("\n")
        if title_re.match(s):                      # a new article starts here
            if cur:
                articles.append("".join(cur))
            cur = [t]
        else:
            cur.append(t)
    if cur:
        articles.append("".join(cur))
    # split by article, not by position within one
    half = len(articles) // 2
    articles = articles[:half] if part == "calib" else articles[half:] if part == "test" else articles
    out = []
    for art in articles:
        ids = tok(art, add_special_tokens=False).input_ids
        if len(ids) >= min_tokens:
            out.append(ids)
        if len(out) >= n_docs:
            break
    if len(out) < n_docs:
        raise RuntimeError(f"not enough independent articles: {len(out)} < {n_docs} (min_tokens={min_tokens})")
    return out


def get_article_splits(tok, min_tokens, splits):
    """Three-way split that is disjoint by article, so selection cannot leak into test.

    splits is a dict such as {"calib": 8, "select": 16, "test": 24}. Articles long enough
    to qualify are ordered by the sha1 of their title and then cut into consecutive
    blocks, which makes the split disjoint by construction, independent of the order the
    corpus happens to be in, and reproducible without storing anything.

    Returns {name: {"ids": [...], "titles": [...], "hashes": [...]}}; the titles and
    hashes are the provenance a reported number can be traced back through.
    """
    import re, hashlib
    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id="Salesforce/wikitext", repo_type="dataset",
                        filename="wikitext-103-raw-v1/test-00000-of-00001.parquet")
    df = pd.read_parquet(p)
    title_re = re.compile(r"^ = [^=].* = $")
    arts, cur, title = [], [], None
    for t in df["text"]:
        s = t.rstrip("\n")
        if title_re.match(s):
            if cur:
                arts.append((title, "".join(cur)))
            cur = [t]; title = s.strip().strip("=").strip()
        else:
            cur.append(t)
    if cur:
        arts.append((title, "".join(cur)))
    quals = []
    for title, art in arts:
        ids = tok(art, add_special_tokens=False).input_ids
        if len(ids) >= min_tokens:
            h = hashlib.sha1((title or "").encode("utf-8")).hexdigest()
            quals.append((h, title, ids))
    quals.sort(key=lambda x: x[0])                         # stable disjoint order by hash
    need = sum(splits.values())
    if len(quals) < need:
        raise RuntimeError(f"not enough qualifying articles: {len(quals)} < {need} (min_tokens={min_tokens})")
    out, i = {}, 0
    for name, k in splits.items():
        chunk = quals[i:i + k]; i += k
        out[name] = dict(ids=[c[2] for c in chunk], titles=[c[1] for c in chunk],
                         hashes=[c[0] for c in chunk])
    return out
