#!/usr/bin/env python3
"""Per-paper harvesting engine, scoped by journal list (FT50 / AJG tiers).

Fetches every paper published by the journals in a chosen scope, matches the model and
concept catalogue (from llm_market_share.py) locally against each title and abstract, and
writes one resumable xlsx workbook. Scope is configuration, not code: `journals.xlsx`
carries each journal's AJG rating and FT50 flag.

Entry points used by the notebooks:
    journals_in_scope(scope)             journals for "ft50" or an AJG rating
    fetch_workbook(scope, path, ...)     harvest -> llm_mentions_<scope>.xlsx, resumable
    flag_matches(df) / compact_review()  false-positive QA

Workbook sheets: `papers` (one row per paper x matched model), `totals` (papers per
journal x year, the denominator), `sources` (journal -> OpenAlex id), `done` (for resume).
"""
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import llm_market_share as M   # MODELS, YEARS, weight_class, MAILTO

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "scripts" else HERE


def _find(fname):
    """Locate a data file regardless of layout: flat (module beside the file), the public
    repo layout (scripts/ + sibling data/), or relative to the working directory."""
    for c in (os.path.join(HERE, fname), os.path.join(HERE, "data", fname),
              os.path.join(ROOT, fname), os.path.join(ROOT, "data", fname),
              os.path.join(os.getcwd(), fname), os.path.join(os.getcwd(), "data", fname)):
        if os.path.exists(c):
            return c
    return os.path.join(ROOT, "data", fname)


JOURNALS = _find("journals.xlsx")
MAILTO = M.MAILTO
SRC_API = "https://api.openalex.org/sources"
WORKS_API = "https://api.openalex.org/works"
START = f"{M.YEARS[0]}-01-01"
END = f"{M.YEARS[-1]}-12-31"

RANK_ORDER = ["4*", "4", "3", "2", "1"]


class QuotaBlocked(Exception):
    def __init__(self, secs): self.secs = secs


# ---------------------------------------------------------------- scope / journals
def journals_in_scope(scope):
    import pandas as pd
    rows = pd.read_excel(_find("journals.xlsx"), dtype=str).fillna("").to_dict("records")
    if scope.lower() == "ft50":
        keep = [r for r in rows if r.get("ft50", "").strip() == "1"]
    else:
        if scope not in RANK_ORDER:
            raise ValueError(f"scope must be 'ft50' or one of {RANK_ORDER} (cumulative)")
        allowed = set(RANK_ORDER[:RANK_ORDER.index(scope) + 1])   # cumulative
        keep = [r for r in rows if r.get("ajg_rank", "").strip() in allowed]
    # de-dup by journal name
    seen, out = set(), []
    for r in keep:
        n = r["journal"].strip()
        if n.lower() not in seen:
            seen.add(n.lower()); out.append(r)
    return out


# ---------------------------------------------------------------- http helper
def _get(url, retries=6):
    url = M._auth(url)                       # append OpenAlex Premium key if set
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": f"llm-share ({MAILTO})"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                ra = int(e.headers.get("Retry-After", 0) or 0)
                if ra > 300:
                    raise QuotaBlocked(ra)
                time.sleep(ra or min(60, 15 * (a + 1)))
            elif a == retries - 1:
                return None
            else:
                time.sleep(2 * (a + 1))
        except Exception:
            if a == retries - 1:
                return None
            time.sleep(2 * (a + 1))


# ---------------------------------------------------------------- resolve journals -> source ids
def _norm(s):
    s = (s or "").lower().replace("&", "and")
    s = re.sub(r"\bthe\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------- model matching
def flex_pattern(name):
    """Match a model name allowing hyphen/space/no-separator variants:
    'GPT-4' also catches 'GPT 4' and 'GPT4'; 'Qwen2.5' catches 'Qwen 2.5'.
    Bounded so 'GPT-4' still does NOT match 'GPT-4o'."""
    toks = re.findall(r'[A-Za-z]+|[0-9]+(?:\.[0-9]+)*', name)   # letter runs / version numbers
    body = r'[\s\-]?'.join(re.escape(t) for t in toks)
    return re.compile(r'(?<![\w.\-])' + body + r'(?![\w.\-])', re.IGNORECASE)


def concept_pattern(name):
    """Looser matcher for umbrella terms: allows an optional trailing plural 's' and
    hyphen/space suffixes. 'LLM' -> LLM, LLMs, LLM-based; 'large language model' ->
    also its plural; 'generative AI' -> also 'generative-AI'. Letter-only boundaries."""
    toks = re.findall(r'[A-Za-z]+', name)
    body = r'[\s\-]?'.join(re.escape(t) for t in toks)
    return re.compile(r'(?<![A-Za-z])' + body + r's?(?![A-Za-z])', re.IGNORECASE)


def build_matchers():
    m = [(q, p, flex_pattern(q)) for q, p, _rel in M.MODELS]
    m += [(c, M.CONCEPT_PROVIDER, concept_pattern(c)) for c in M.CONCEPTS]
    return m


# ---------------------------------------------------------------- false-positive QA
# Matched names that also carry an everyday meaning -> route to human/LLM review. Kept to
# terms where the non-model sense is genuinely competitive in business/econ abstracts.
# Deliberately NOT included: 'BERT'/'Gemma' (huge true-mention base rate, so flagging them
# is mostly noise; the 'Bert'/'Gemma' person sense is rare) -- add them to over-review.
AMBIGUOUS = {
    "LLM",         # = Master of Laws ("LL.M.") degree (law/tax/accounting journals)
    "Vicuna",      # = camelid / wool
    "Chinchilla",  # = animal / fur
    "Orca",        # = whale
    "Gopher",      # = animal / old internet protocol
}


def _snippet(text, pat, width=55):
    """A short context window around the first match, for eyeballing a row."""
    if not (text and pat):
        return ""
    mm = pat.search(text)
    if not mm:
        return ""
    a, b = max(0, mm.start() - width), min(len(text), mm.end() + width)
    return ("…" if a else "") + " ".join(text[a:b].split()) + ("…" if b < len(text) else "")


def flag_matches(df):
    """Add QA columns to a papers dataframe for false-positive review (NO API calls).

        pre_release  paper year < the model's release year -> temporally impossible
                     (e.g. a Cohere Command mention in a 2020-2021 paper)
        stale_match  the CURRENT matcher no longer fires on the stored title/abstract
                     -> the row was left by an outdated/removed rule (old bare 'Command A',
                        dropped 'PaLM', ...)
        ambiguous    matched term is on the AMBIGUOUS watchlist (e.g. 'LLM' = law degree)
        snippet      text around the match, for a quick human read
        flag         comma-joined active flags ('' = looks clean)

    Use it on the workbook's `papers` sheet to build a review list, then drop or keep
    rows as you see fit. Because it re-checks stored text, it retro-cleans data fetched
    under older rules without re-downloading anything.
    """
    import pandas as pd
    rel = {q: r for q, _p, r in M.MODELS}
    pats = {q: flex_pattern(q) for q, _p, _r in M.MODELS}
    pats.update({c: concept_pattern(c) for c in M.CONCEPTS})

    def _one(r):
        q = r.get("model")
        pat = pats.get(q)
        title, ab = str(r.get("title") or ""), str(r.get("abstract") or "")
        yr = str(r.get("year") or "")
        stale = not (pat and (pat.search(title) or pat.search(ab)))
        pre = (q in rel) and yr.isdigit() and int(yr) < rel[q]
        amb = q in AMBIGUOUS
        snip = _snippet(title, pat) or _snippet(ab, pat)
        flag = ",".join(k for k, v in
                        [("pre_release", pre), ("stale_match", stale), ("ambiguous", amb)] if v)
        return pd.Series({"pre_release": pre, "stale_match": stale,
                          "ambiguous": amb, "snippet": snip, "flag": flag})

    add = df.apply(_one, axis=1)
    return pd.concat([df.reset_index(drop=True), add.reset_index(drop=True)], axis=1)


def fetch_workbook(scope, wb_path, sleep=0.2, start_date=None, end_date=None):
    """Per-paper harvest for `scope`, persisted ENTIRELY to one xlsx workbook
    (sheets papers/totals/sources/done). Resumable via the 'done' sheet; writes NO CSVs.
    Snapshot-saves every 3 journals via an atomic os.replace. Needs pandas + openpyxl."""
    import pandas as pd
    wb_path = str(wb_path)
    start_date = start_date or START; end_date = end_date or END   # OpenAlex pub-date window
    illegal = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
    clean = lambda s: illegal.sub("", str(s))[:32000]
    PCOLS = ["openalex_id", "journal", "issn", "tier", "ft50", "year", "month",
             "model", "provider", "weights", "location", "title", "abstract"]
    TCOLS = ["source_id", "journal", "tier", "year", "papers"]
    SCOLS = ["journal", "source_id", "resolved_name", "issn_l", "match"]
    os.makedirs(os.path.dirname(wb_path) or ".", exist_ok=True)

    papers, totals, srccache, done = [], [], {}, set()
    if os.path.exists(wb_path):
        xl = pd.read_excel(wb_path, sheet_name=None)
        papers = xl["papers"].fillna("").values.tolist() if "papers" in xl else []
        totals = xl["totals"].fillna("").values.tolist() if "totals" in xl else []
        if "sources" in xl:
            for r in xl["sources"].fillna("").to_dict("records"):
                srccache[r["journal"]] = r
        if "done" in xl and "source_id" in xl["done"].columns:
            done = set(xl["done"]["source_id"].astype(str))

    def save():
        tmp = wb_path[:-5] + ".tmp.xlsx"
        with pd.ExcelWriter(tmp, engine="openpyxl") as xw:
            pd.DataFrame(papers, columns=PCOLS).to_excel(xw, sheet_name="papers", index=False)
            pd.DataFrame(totals, columns=TCOLS).to_excel(xw, sheet_name="totals", index=False)
            sdf = pd.DataFrame(list(srccache.values()), columns=SCOLS) if srccache else pd.DataFrame(columns=SCOLS)
            sdf.to_excel(xw, sheet_name="sources", index=False)
            pd.DataFrame({"source_id": sorted(done)}).to_excel(xw, sheet_name="done", index=False)
        os.replace(tmp, wb_path)

    journals = journals_in_scope(scope)
    tier_of = {j["journal"]: (j.get("ajg_rank", ""), j.get("ft50", "")) for j in journals}

    for j in journals:                                  # resolve journals -> source ids
        if j["journal"] in srccache:
            continue
        name = j["journal"]; issn = (j.get("issn") or "").strip()
        chosen, match = None, "notfound"
        if issn:
            d = _get(SRC_API + "?" + urllib.parse.urlencode(
                {"filter": f"issn:{issn}", "per-page": 1, "select": "id,display_name,issn_l", "mailto": MAILTO}))
            res = (d or {}).get("results") or []
            if res:
                chosen, match = res[0], "issn"
        if chosen is None:
            d = _get(SRC_API + "?" + urllib.parse.urlencode(
                {"search": name, "per-page": 25, "select": "id,display_name,issn_l", "mailto": MAILTO}))
            res = (d or {}).get("results") or []
            tgt = _norm(name); exact = [s for s in res if _norm(s.get("display_name", "")) == tgt]
            if exact:
                chosen, match = exact[0], "exact"
            elif res:
                chosen, match = res[0], "FUZZY"
        srccache[name] = ({"journal": name, "source_id": chosen["id"].rsplit("/", 1)[-1],
                           "resolved_name": chosen.get("display_name", ""),
                           "issn_l": chosen.get("issn_l", ""), "match": match} if chosen else
                          {"journal": name, "source_id": "", "resolved_name": "NOT FOUND",
                           "issn_l": "", "match": "notfound"})
        time.sleep(sleep)
    save()

    sources = {}
    for j in journals:
        info = srccache.get(j["journal"], {})
        sid = info.get("source_id", "")
        if not sid or info.get("match") in ("FUZZY", "notfound"):
            continue
        sources[sid] = (info.get("resolved_name") or j["journal"], *tier_of[j["journal"]], info.get("issn_l", ""))
    todo = [s for s in sources if s not in done]
    print(f"scope '{scope}': {len(journals)} journals | {len(todo)} sources to fetch ({len(done)} done)")

    matchers = build_matchers()
    try:
        for si, sid in enumerate(todo, 1):
            jname, tier, ft50, issn = sources[sid]
            ytot = {}; cursor = "*"; npapers = 0
            while cursor:
                d = _get(WORKS_API + "?" + urllib.parse.urlencode(
                    {"filter": f"primary_location.source.id:{sid},from_publication_date:{start_date},to_publication_date:{end_date}",
                     "per-page": 200, "cursor": cursor,
                     "select": "id,title,abstract_inverted_index,publication_date", "mailto": MAILTO}))
                if d is None:
                    break
                for wk in d.get("results", []):
                    date = wk.get("publication_date") or ""
                    yr = int(date[:4]) if date[:4].isdigit() else None
                    if yr is None or yr not in M.YEARS:
                        continue
                    mo = int(date[5:7]) if len(date) >= 7 and date[5:7].isdigit() else ""
                    ytot[yr] = ytot.get(yr, 0) + 1; npapers += 1
                    title = wk.get("title") or ""
                    ab = abstract_from_inverted(wk.get("abstract_inverted_index"))
                    for q, prov, pat in matchers:
                        loc = "title" if pat.search(title) else ("abstract" if pat.search(ab) else None)
                        if loc:
                            papers.append([wk["id"].rsplit("/", 1)[-1], jname, issn, tier, ft50, yr, mo,
                                           q, prov, M.weight_class(q, prov), loc, clean(title), clean(ab)])
                cursor = d["meta"].get("next_cursor")
                if not d.get("results"):
                    break
                time.sleep(sleep)
            for y, n in sorted(ytot.items()):
                totals.append([sid, jname, tier, y, n])
            done.add(sid)
            if si % 3 == 0 or si == len(todo):
                save(); print(f"  {si}/{len(todo)}  {jname}: {npapers} papers")
    except QuotaBlocked as e:
        print(f"STOPPED: rate-limited (~{e.secs/3600:.1f}h). Saved; re-run to resume.")
    save()
    print(f"workbook -> {wb_path}")


def compact_review(flagged, drop_terms=("LLM",), max_rows=None):
    """Small, paste-ready slice of flag_matches() output for a chat-based review:
    keeps pre_release + stale_match + (ambiguous minus high-volume `drop_terms`, e.g.
    'LLM' whose true-mention base rate would flood the list), and drops the bulky
    title/abstract (the `snippet` carries the context). Paste it straight into a chat."""
    keep = (flagged["pre_release"] | flagged["stale_match"]
            | (flagged["ambiguous"] & ~flagged["model"].isin(list(drop_terms))))
    cols = ["openalex_id", "year", "journal", "model", "provider", "flag", "snippet"]
    out = flagged.loc[keep, cols].sort_values(["flag", "model", "year"]).reset_index(drop=True)
    return out.head(max_rows) if max_rows else out


def abstract_from_inverted(inv):
    if not inv:
        return ""
    pos = [(i, w) for w, idxs in inv.items() for i in idxs]
    pos.sort()
    return " ".join(w for _, w in pos)


# ---------------------------------------------------------------- works fetch


