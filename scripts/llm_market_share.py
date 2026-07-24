#!/usr/bin/env python3
"""
LLM "market share" via academic mentions (OpenAlex).

IDEA
----
OpenAlex (https://openalex.org) indexes ~250M scholarly works and returns a
`meta.count` for *any* filtered query. So instead of downloading millions of
papers, we ask it one cheap question per (model, year):

    "How many works published in <year> mention <model>?"

Summed by provider, the yearly counts give a rough proxy for each LLM provider's
*mindshare in the research literature* over time.

IMPORTANT CAVEAT (read before believing the numbers)
----------------------------------------------------
This measures how often a model is NAMED IN PAPERS, which is not the same as
commercial market share:
  * open-weight models (Llama, Qwen, Mistral, DeepSeek) are over-represented in
    research relative to their commercial footprint, because researchers can run
    and fine-tune them;
  * closed API usage (a lot of ChatGPT/Claude/Gemini traffic) is invisible here;
  * name collisions inflate common words (Bard, PaLM, Grok, Nova, Phi, Llama,
    Gemini, Claude...). We disambiguate with versioned / qualified phrases, but
    some noise remains.
Treat it as "academic attention share", a directional proxy, not ground truth.

USAGE
-----
    python llm_market_share.py            # full run (cached; safe to re-run)
    python llm_market_share.py --test     # quick sanity run (a few models/years)
    python llm_market_share.py --refresh  # ignore cache and re-fetch everything

Outputs (in ./data and ./figures):
    data/mentions_raw.xlsx      long table: model, provider, query, year, count
    data/provider_share.xlsx    provider x year counts and shares
    figures/provider_share.pdf  stacked-area share over time
    figures/provider_counts.pdf absolute mentions over time (log y)
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figures")
RAW_XLSX = os.path.join(DATA, "mentions_raw.xlsx")
SHARE_XLSX = os.path.join(DATA, "provider_share.xlsx")

MAILTO = "guillaume.coqueret@gmail.com"     # OpenAlex "polite pool" -> faster, nicer
# OpenAlex Premium key. Read from the environment so it is NEVER hard-coded in a file
# that might be shared/committed. Set it with:  export OPENALEX_API_KEY="your-key"
# (or, in a notebook, assign  M.API_KEY = "your-key"  before fetching). When set, the
# rate limit attaches to the KEY, not your IP -> no more shared-IP 429 bans on Colab.
API_KEY = os.environ.get("OPENALEX_API_KEY", "").strip()
BASE = "https://api.openalex.org/works"


def _auth(url):
    """Append the OpenAlex premium api_key (if API_KEY is set) to a fully-built URL."""
    if not API_KEY:
        return url
    sep = "&" if "?" in url else "?"
    return url + sep + "api_key=" + urllib.parse.quote(API_KEY)
YEARS = list(range(2020, 2027))             # 2020..2026 (incl. pre-ChatGPT baseline: BERT-era + "LLM"/"gen AI" terms predate 2022)

# Where a model name is looked for. "any" = title+abstract+fulltext (the deduplicated
# union, used for shares); the other three are location breakdowns and OVERLAP each other.
SEARCH_FIELDS = {"title": "title.search", "abstract": "abstract.search",
                 "fulltext": "fulltext.search"}
LOCATIONS = ["any", "title", "abstract", "fulltext"]

# ---------------------------------------------------------------------------
# Model catalogue: (search_query, provider, release_year)
# search_query is matched as an EXACT PHRASE (quotes added automatically).
# Queries are chosen to be as unambiguous as possible; ambiguous bare names
# (Bard, PaLM, Grok, Nova, Llama, Claude, Gemini...) are given versioned or
# qualified forms. Edit freely.
# ---------------------------------------------------------------------------
# NOTE on ambiguous names: some model names collide with common words and are
# hard to count cleanly (Yi, Spark, Step, MiniMax [=game-theory algorithm], ABAB,
# Nova, Titan, Command, Aya, Grok, Orca, Dolly, Koala, Zephyr, BLOOM, OPT, Falcon,
# Skylark, PanGu). We qualify these where possible; residual noise is unavoidable.
# Pure pre-LLM encoders (BERT, RoBERTa, T5) are intentionally EXCLUDED (not generative).
MODELS = [
    # ===================== OpenAI =====================
    ("GPT-2", "OpenAI", 2019), ("GPT-3", "OpenAI", 2020), ("InstructGPT", "OpenAI", 2022),
    ("GPT-3.5", "OpenAI", 2022), ("ChatGPT", "OpenAI", 2022), ("GPT-4", "OpenAI", 2023),
    ("GPT-4 Turbo", "OpenAI", 2023), ("GPT-4V", "OpenAI", 2023), ("GPT-4o", "OpenAI", 2024),
    ("GPT-4o mini", "OpenAI", 2024), ("GPT-4.1", "OpenAI", 2025), ("GPT-4.5", "OpenAI", 2025),
    ("OpenAI o1", "OpenAI", 2024), ("o1-mini", "OpenAI", 2024), ("OpenAI o3", "OpenAI", 2025),
    ("o3-mini", "OpenAI", 2025), ("o4-mini", "OpenAI", 2025), ("GPT-5", "OpenAI", 2025),
    ("GPT-5.1", "OpenAI", 2025), ("GPT-5.2", "OpenAI", 2025),
    ("GPT-5.4", "OpenAI", 2026), ("GPT-5.5", "OpenAI", 2026), ("GPT-5.6", "OpenAI", 2026),
    # ===================== Anthropic =====================
    ("Claude 1", "Anthropic", 2023), ("Claude 2", "Anthropic", 2023),
    ("Claude Instant", "Anthropic", 2023), ("Claude 3", "Anthropic", 2024),
    ("Claude 3 Opus", "Anthropic", 2024), ("Claude 3 Sonnet", "Anthropic", 2024),
    ("Claude 3 Haiku", "Anthropic", 2024), ("Claude 3.5 Sonnet", "Anthropic", 2024),
    ("Claude 3.5 Haiku", "Anthropic", 2024), ("Claude 3.7 Sonnet", "Anthropic", 2025),
    ("Claude Opus 4", "Anthropic", 2025), ("Claude Sonnet 4", "Anthropic", 2025),
    ("Claude Opus 4.1", "Anthropic", 2025), ("Claude Sonnet 4.5", "Anthropic", 2025),
    ("Claude Haiku 4.5", "Anthropic", 2025), ("Claude Opus 4.5", "Anthropic", 2025),
    ("Claude Opus 4.6", "Anthropic", 2026), ("Claude Opus 4.7", "Anthropic", 2026),
    ("Claude Opus 4.8", "Anthropic", 2026), ("Claude Sonnet 4.6", "Anthropic", 2026),
    ("Claude Sonnet 5", "Anthropic", 2026), ("Claude Fable 5", "Anthropic", 2026),
    # ===================== Google / DeepMind =====================
    ("Gopher", "Google", 2021), ("Chinchilla", "Google", 2022), ("LaMDA", "Google", 2022),
    ("Flan-T5", "Google", 2022), ("PaLM 2", "Google", 2023),   # bare "PaLM" dropped (matches "palm oil")
    ("Med-PaLM", "Google", 2023), ("Med-PaLM 2", "Google", 2023), ("Google Bard", "Google", 2023),
    ("Gemini 1.0", "Google", 2023), ("Gemini 1.5", "Google", 2024), ("Gemini 2.0", "Google", 2024),
    ("Gemini 2.5", "Google", 2025), ("Gemini 3", "Google", 2025), ("Gemini 3.5", "Google", 2026),
    ("Gemini 3.6", "Google", 2026), ("Gemma", "Google", 2024), ("Gemma 2", "Google", 2024),
    ("Gemma 3", "Google", 2025), ("Gemma 4", "Google", 2026), ("CodeGemma", "Google", 2024),
    # ===================== Meta =====================
    ("OPT-175B", "Meta", 2022), ("Galactica", "Meta", 2022), ("LLaMA", "Meta", 2023),
    ("Llama 2", "Meta", 2023), ("Code Llama", "Meta", 2023), ("Llama 3", "Meta", 2024),
    ("Llama 3.1", "Meta", 2024), ("Llama 3.2", "Meta", 2024), ("Llama 3.3", "Meta", 2024),
    ("Llama 4", "Meta", 2025), ("Muse Spark", "Meta", 2026),
    # ===================== Microsoft =====================
    ("Phi-1", "Microsoft", 2023), ("Phi-2", "Microsoft", 2023), ("Phi-3", "Microsoft", 2024),
    ("Phi-3.5", "Microsoft", 2024), ("Phi-4", "Microsoft", 2024), ("WizardLM", "Microsoft", 2023),
    ("Orca", "Microsoft", 2023),
    # ===================== xAI =====================
    ("Grok-1", "xAI", 2023), ("Grok-1.5", "xAI", 2024), ("Grok 2", "xAI", 2024),
    ("Grok 3", "xAI", 2025), ("Grok 4", "xAI", 2025), ("Grok 4.5", "xAI", 2026),
    # ===================== Mistral (full lineup) =====================
    ("Mistral 7B", "Mistral", 2023), ("Mixtral 8x7B", "Mistral", 2023),
    ("Mixtral 8x22B", "Mistral", 2024), ("Mistral Large", "Mistral", 2024),
    ("Mistral Large 2", "Mistral", 2024), ("Mistral Small", "Mistral", 2024),
    ("Mistral Medium", "Mistral", 2024), ("Mistral Nemo", "Mistral", 2024),
    ("Codestral", "Mistral", 2024), ("Mathstral", "Mistral", 2024), ("Pixtral", "Mistral", 2024),
    ("Ministral", "Mistral", 2024), ("Mistral Small 3", "Mistral", 2025),
    ("Mistral Medium 3", "Mistral", 2025), ("Magistral", "Mistral", 2025),
    ("Mistral Large 3", "Mistral", 2025), ("Mistral Small 4", "Mistral", 2026),
    ("Mistral Medium 3.5", "Mistral", 2026), ("Ministral 3", "Mistral", 2025),
    # ===================== Other Western / open =====================
    # Command models qualified with the "Cohere" brand: bare "Command A" collides with
    # "command a premium", "Command R" with "command R&D" -- common in business abstracts.
    ("Cohere Command", "Cohere", 2023), ("Cohere Command R", "Cohere", 2024),
    ("Cohere Command R+", "Cohere", 2024), ("Cohere Command A", "Cohere", 2025), ("Cohere Aya", "Cohere", 2024),
    ("Amazon Titan", "Amazon", 2023), ("Amazon Nova", "Amazon", 2024),
    ("Jurassic-2", "AI21", 2023), ("Jamba", "AI21", 2024),
    ("Nemotron", "NVIDIA", 2024), ("Megatron-LM", "NVIDIA", 2019),
    ("DBRX", "Databricks", 2024), ("MPT-7B", "MosaicML", 2023),
    ("Falcon-40B", "TII", 2023), ("Falcon-180B", "TII", 2023),
    ("GPT-NeoX", "EleutherAI", 2022), ("GPT-J", "EleutherAI", 2021), ("Pythia", "EleutherAI", 2023),
    ("BLOOM language model", "BigScience", 2022), ("StableLM", "Stability AI", 2023),
    ("Vicuna", "LMSYS/Academic", 2023), ("Alpaca language model", "Stanford/Academic", 2023),
    ("StarCoder", "BigCode", 2023),
    # ===================== CHINA: Alibaba (Qwen / Tongyi) =====================
    ("Qwen", "Alibaba", 2023), ("Qwen-7B", "Alibaba", 2023), ("Qwen-VL", "Alibaba", 2023),
    ("Qwen1.5", "Alibaba", 2024), ("Qwen2", "Alibaba", 2024), ("Qwen2.5", "Alibaba", 2024),
    ("QwQ", "Alibaba", 2024), ("Qwen2.5-Max", "Alibaba", 2025), ("Qwen3", "Alibaba", 2025),
    ("Qwen 3.5", "Alibaba", 2026), ("Tongyi Qianwen", "Alibaba", 2023),
    # ===================== CHINA: DeepSeek =====================
    ("DeepSeek LLM", "DeepSeek", 2023), ("DeepSeek-Coder", "DeepSeek", 2023),
    ("DeepSeek-V2", "DeepSeek", 2024), ("DeepSeek-V3", "DeepSeek", 2024),
    ("DeepSeek-VL", "DeepSeek", 2024), ("DeepSeek-Math", "DeepSeek", 2024),
    ("DeepSeek-R1", "DeepSeek", 2025), ("DeepSeek-V3.1", "DeepSeek", 2025),
    ("DeepSeek-V3.2", "DeepSeek", 2025), ("DeepSeek-V4", "DeepSeek", 2026),
    # ===================== CHINA: Zhipu AI / THUDM (GLM) =====================
    ("GLM-130B", "Zhipu AI", 2022), ("ChatGLM", "Zhipu AI", 2023), ("ChatGLM2", "Zhipu AI", 2023),
    ("ChatGLM3", "Zhipu AI", 2023), ("GLM-4", "Zhipu AI", 2024), ("GLM-4.5", "Zhipu AI", 2025),
    ("GLM-5", "Zhipu AI", 2026), ("GLM-5.2", "Zhipu AI", 2026), ("CogVLM", "Zhipu AI", 2023),
    # ===================== CHINA: Moonshot AI (Kimi) =====================
    ("Kimi Moonshot", "Moonshot AI", 2023), ("Kimi K1.5", "Moonshot AI", 2025),
    ("Kimi K2", "Moonshot AI", 2025), ("Kimi K2.5", "Moonshot AI", 2026),
    # ===================== CHINA: 01.AI (Yi) =====================
    ("Yi-34B", "01.AI", 2023), ("Yi-1.5", "01.AI", 2024), ("Yi-Large", "01.AI", 2024),
    # ===================== CHINA: Baidu (ERNIE / Wenxin) =====================
    ("ERNIE 3.0", "Baidu", 2021), ("ERNIE Bot", "Baidu", 2023), ("ERNIE 4.0", "Baidu", 2023),
    ("ERNIE 4.5", "Baidu", 2025), ("ERNIE 5.0", "Baidu", 2026), ("ERNIE 5.1", "Baidu", 2026),
    ("Wenxin Yiyan", "Baidu", 2023),
    # ===================== CHINA: ByteDance (Doubao) =====================
    ("Doubao", "ByteDance", 2024), ("Doubao 2.0", "ByteDance", 2026), ("ByteDance Skylark", "ByteDance", 2023),
    # ===================== CHINA: Tencent (Hunyuan) =====================
    ("Hunyuan", "Tencent", 2023), ("Hunyuan-Large", "Tencent", 2024), ("Hunyuan-T1", "Tencent", 2026),
    # ===================== CHINA: iFlytek (Spark) =====================
    ("iFlytek Spark", "iFlytek", 2023), ("SparkDesk", "iFlytek", 2023),
    # ===================== CHINA: MiniMax =====================
    ("MiniMax-Text", "MiniMax", 2024), ("MiniMax-01", "MiniMax", 2025), ("MiniMax M3", "MiniMax", 2026),
    # ===================== CHINA: Baichuan =====================
    ("Baichuan", "Baichuan", 2023), ("Baichuan2", "Baichuan", 2023),
    # ===================== CHINA: InternLM (Shanghai AI Lab) =====================
    ("InternLM", "Shanghai AI Lab", 2023), ("InternLM2", "Shanghai AI Lab", 2024),
    ("InternLM2.5", "Shanghai AI Lab", 2024),
    # ===================== CHINA: SenseTime =====================
    ("SenseChat", "SenseTime", 2023), ("SenseNova", "SenseTime", 2023),
    # ===================== CHINA: others =====================
    ("StepFun", "StepFun", 2024), ("Skywork", "Kunlun", 2023),
    ("Huawei PanGu", "Huawei", 2021), ("Yuan 2.0", "Inspur", 2023),
    # ===================== ENCODER-ERA (BERT family) =====================
    # Encoder-only masked-LMs, pre-generative (2018-2020) but still heavily used as
    # baselines/embeddings in 2022-2026 papers. All open-weights. Kept as ordinary
    # model rows; how (or whether) to fold them into generative shares is a
    # downstream analysis choice.
    ("BERT", "Google", 2018), ("RoBERTa", "Meta", 2019),
    ("DistilBERT", "Hugging Face", 2019), ("ALBERT", "Google", 2019),
    ("ELECTRA", "Google", 2020), ("DeBERTa", "Microsoft", 2020),
    ("XLNet", "Google", 2019), ("mBERT", "Google", 2018),
    ("XLM-R", "Meta", 2019), ("XLM-RoBERTa", "Meta", 2019),
    ("BioBERT", "Academic", 2019), ("SciBERT", "Allen Institute for AI", 2019),
    ("FinBERT", "Academic", 2019), ("ClinicalBERT", "Academic", 2019),
    ("Legal-BERT", "Academic", 2020),
]

# a reference denominator: all works mentioning "large language model"
REFERENCE_QUERIES = [("large language model", "(reference: LLM literature)", 2019)]

# Umbrella / topic-level search terms -- NOT models (no provider, no weights). These
# capture how often the literature uses generic generative-AI vocabulary, i.e. an
# adoption/denominator signal alongside the per-model rows. Matched per paper with a
# LOOSER pattern than models (concept_pattern): catches plurals and hyphen suffixes
# ("LLMs", "LLM-based", "generative-AI"). Tagged provider=CONCEPT_PROVIDER so they can
# never leak into provider/open-vs-closed shares.
CONCEPT_PROVIDER = "(concept)"
CONCEPTS = [
    "large language model",                 # + "large language models"
    "LLM",                                  # + "LLMs", "LLM-based"
    "generative AI",                        # + "generative-AI"
    "generative artificial intelligence",
    "gen AI",                               # + "GenAI", "gen-AI"
    "foundation model",                     # + "foundation models"
]

# ---------------------------------------------------------------------------
# Open- vs closed-weights classification.
# "open" = weights publicly downloadable (any license); "closed" = API-only.
# Reflects each model's best-known/flagship status; a few are genuinely mixed
# or evolving (noted). Providers whose lineup is essentially all-open are listed;
# everything else defaults to closed, with per-model overrides for exceptions.
# ---------------------------------------------------------------------------
_OPEN_PROVIDERS = {
    "Meta", "Mistral", "Microsoft", "Alibaba", "DeepSeek", "Zhipu AI", "01.AI",
    "Baichuan", "Shanghai AI Lab", "EleutherAI", "BigScience", "Stability AI",
    "MosaicML", "Databricks", "NVIDIA", "TII", "LMSYS/Academic",
    "Stanford/Academic", "BigCode", "Kunlun", "Inspur",
}
_WEIGHT_OVERRIDE = {
    # exceptions to the provider default
    "GPT-2": "open",                                             # OpenAI (else closed)
    "Gemma": "open", "Gemma 2": "open", "Gemma 3": "open",       # Google (else closed)
    "CodeGemma": "open", "Flan-T5": "open",
    "Grok-1": "open", "Grok 2": "open",                          # xAI (else closed)
    "Cohere Command R": "open", "Cohere Command R+": "open", "Cohere Command A": "open", "Cohere Aya": "open",
    "Jamba": "open",                                             # AI21 (else closed)
    "Kimi K2": "open",                                           # Moonshot (else closed)
    "ERNIE 4.5": "open",                                         # Baidu (else closed)
    "Hunyuan-Large": "open",                                     # Tencent (else closed)
    "MiniMax-01": "open", "MiniMax-Text": "open",               # MiniMax (else closed)
    # open-provider exceptions that are actually API-only / unreleased
    "Mistral Large": "closed", "Mistral Medium": "closed", "Mistral Medium 3": "closed",
    "Yi-Large": "closed",                                        # 01.AI (else open)
    "Orca": "closed",                                            # Microsoft (weights unreleased)
    # BERT family -- all open-weights (providers not in _OPEN_PROVIDERS, so set explicitly)
    "BERT": "open", "DistilBERT": "open", "ALBERT": "open", "ELECTRA": "open",
    "XLNet": "open", "mBERT": "open", "BioBERT": "open", "SciBERT": "open",
    "FinBERT": "open", "ClinicalBERT": "open", "Legal-BERT": "open",
    # 2026 additions whose weights differ from the provider default
    "Gemma 4": "open", "Muse Spark": "closed", "Mistral Medium 3.5": "closed",
    "Kimi K2.5": "open", "MiniMax M3": "open", "Hunyuan-T1": "open",
}


def weight_class(query, provider):
    """'open' or 'closed' weights for a model query ('n/a' for umbrella concepts)."""
    if provider == CONCEPT_PROVIDER:
        return "n/a"
    if query in _WEIGHT_OVERRIDE:
        return _WEIGHT_OVERRIDE[query]
    return "open" if provider in _OPEN_PROVIDERS else "closed"


class QuotaBlocked(Exception):
    """Raised when OpenAlex returns a long (daily-scale) rate-limit ban."""
    def __init__(self, secs):
        self.secs = secs


def openalex_count(query, year, location="any", retries=7):
    """Number of OpenAlex works from `year` mentioning the exact phrase `query`.
    location: 'any' (title+abstract+fulltext union) or 'title'/'abstract'/'fulltext'."""
    if location == "any":
        params = {"search": f'"{query}"', "filter": f"publication_year:{year}",
                  "per-page": 1, "select": "id", "mailto": MAILTO}
    else:
        params = {"filter": f'{SEARCH_FIELDS[location]}:"{query}",publication_year:{year}',
                  "per-page": 1, "select": "id", "mailto": MAILTO}
    url = _auth(BASE + "?" + urllib.parse.urlencode(params))
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": f"llm-market-share ({MAILTO})"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)["meta"]["count"]
        except urllib.error.HTTPError as e:
            if e.code == 429:                     # rate limited
                try:
                    ra = int(e.headers.get("Retry-After", 0))
                except (TypeError, ValueError):
                    ra = 0
                if ra > 300:                      # daily-scale ban: abort fast, don't hang
                    raise QuotaBlocked(ra)
                wait = ra if ra else min(90, 15 * (attempt + 1))
                print(f"  .. transient 429 on {query!r} {year}; waiting {wait}s", file=sys.stderr)
                time.sleep(wait)
            elif attempt == retries - 1:
                print(f"  ! failed {query!r} {year}: {e}", file=sys.stderr)
                return None
            else:
                time.sleep(2 * (attempt + 1))
        except Exception as e:                    # noqa: BLE001 (retry anything transient)
            if attempt == retries - 1:
                print(f"  ! failed {query!r} {year}: {e}", file=sys.stderr)
                return None
            time.sleep(2 * (attempt + 1))


def load_cache():
    cache = {}
    if os.path.exists(RAW_XLSX):
        import pandas as pd
        for row in pd.read_excel(RAW_XLSX).fillna("").to_dict("records"):
            try:
                cache[(row["query"], int(row["year"]), row.get("location", "any"))] = row
            except (ValueError, TypeError, KeyError):
                continue
    return cache


def _save_raw(rows):
    """Atomic-rewrite the mention cache to xlsx (no CSV). Called periodically so a crash
    loses at most the last save-interval of cells rather than a single row."""
    import pandas as pd
    os.makedirs(DATA, exist_ok=True)
    cols = ["model", "provider", "weights", "query", "year", "location", "count"]
    tmp = RAW_XLSX[:-5] + ".tmp.xlsx"
    pd.DataFrame(rows, columns=cols).to_excel(tmp, index=False)
    os.replace(tmp, RAW_XLSX)


def fetch(models, years, refresh=False, sleep=1.0):
    """Fetch missing (model, year, location) cells, snapshot-saving the xlsx cache every
    40 cells (+ on exit) via an atomic os.replace. Resumes from the saved workbook."""
    os.makedirs(DATA, exist_ok=True)
    if refresh and os.path.exists(RAW_XLSX):
        os.remove(RAW_XLSX)

    def _valid(r):
        try:
            int(r["count"]); return True
        except (ValueError, TypeError):
            return False

    rows = [r for r in load_cache().values() if _valid(r)] if not refresh else []
    have = {(r["query"], int(r["year"]), r.get("location", "any")) for r in rows}
    todo = [(m, y, loc) for m in models for y in years if y >= m[2]
            for loc in LOCATIONS if (m[0], y, loc) not in have]
    print(f"{len(todo)} (model,year,location) cells to fetch ({len(have)} already saved on disk)...")

    try:
        for i, ((query, provider, _rel), year, loc) in enumerate(todo, 1):
            try:
                c = openalex_count(query, year, location=loc)
            except QuotaBlocked as e:
                print(f"\nSTOPPED: OpenAlex quota exhausted (resets in ~{e.secs/3600:.1f}h). "
                      f"{len(have)} cells saved to xlsx; re-run to resume (or use a different IP).")
                break
            if c is None:
                continue
            rows.append({"model": query, "provider": provider,
                         "weights": weight_class(query, provider), "query": query,
                         "year": year, "location": loc, "count": c})
            have.add((query, year, loc))
            if i % 40 == 0 or i == len(todo):
                _save_raw(rows)
                print(f"  {i}/{len(todo)}  last: {query} {year} [{loc}] -> {c}")
            time.sleep(sleep)
    finally:
        _save_raw(rows)
    return rows


def aggregate(rows):
    """provider -> {year -> total count}. Uses the current MODELS catalogue as the
    authoritative query->provider map, so stale/removed rows in the cache are ignored."""
    provider_of = {q: p for q, p, _ in MODELS}
    prov = {}
    for r in rows:
        q = r["query"]
        if q not in provider_of:               # skip reference rows and stale cache entries
            continue
        if r.get("location", "any") != "any":  # shares use the deduplicated union only
            continue
        try:
            c = int(r["count"])
        except (ValueError, TypeError):
            continue
        p = provider_of[q]
        prov.setdefault(p, {}).setdefault(int(r["year"]), 0)
        prov[p][int(r["year"])] += c
    return prov


def write_shares(prov, years):
    import pandas as pd
    os.makedirs(DATA, exist_ok=True)
    providers = sorted(prov, key=lambda p: -sum(prov[p].values()))
    totals = {y: sum(prov[p].get(y, 0) for p in providers) for y in years}
    data = []
    for p in providers:
        row = {"provider": p}
        row.update({f"count_{y}": prov[p].get(y, 0) for y in years})
        row.update({f"share_{y}": round(prov[p].get(y, 0) / totals[y], 4) if totals[y] else 0 for y in years})
        data.append(row)
    pd.DataFrame(data).to_excel(SHARE_XLSX, index=False)
    return providers, totals


def plot(prov, providers, years):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                          # noqa: BLE001
        print(f"(matplotlib unavailable, skipping plots: {e})")
        return
    os.makedirs(FIGS, exist_ok=True)
    totals = {y: sum(prov[p].get(y, 0) for p in providers) for y in years}

    # 1) stacked-area share
    fig, ax = plt.subplots(figsize=(9, 5))
    shares = [[prov[p].get(y, 0) / totals[y] if totals[y] else 0 for y in years] for p in providers]
    ax.stackplot(years, shares, labels=providers)
    ax.set_ylim(0, 1); ax.set_xlim(min(years), max(years))
    ax.set_ylabel("share of tracked LLM mentions"); ax.set_xlabel("publication year")
    ax.set_title("LLM provider mindshare in the research literature (OpenAlex mentions)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "provider_share.pdf"))

    # 2) absolute counts (log)
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for p in providers:
        ax2.plot(years, [prov[p].get(y, 0) for y in years], marker="o", label=p)
    ax2.set_yscale("log"); ax2.set_ylabel("works mentioning provider's models")
    ax2.set_xlabel("publication year")
    ax2.set_title("Academic mentions per LLM provider per year (OpenAlex)")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig2.tight_layout(); fig2.savefig(os.path.join(FIGS, "provider_counts.pdf"))
    print(f"figures -> {FIGS}/provider_share.pdf, provider_counts.pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", action="store_true", help="quick run: few models, 2024-2025 only")
    ap.add_argument("--refresh", action="store_true", help="ignore cache, re-fetch all")
    ap.add_argument("--slow", action="store_true",
                    help="gentler pace (1 request / 2.5s) to avoid rate-limit bans")
    ap.add_argument("--fast", action="store_true",
                    help="~3 requests/s (0.3s) -- safe on a dedicated IP (cloud VM); still under OpenAlex's 10/s")
    ap.add_argument("--api-key", default="", help="OpenAlex Premium key (overrides $OPENALEX_API_KEY)")
    args = ap.parse_args()
    if args.api_key:
        globals()["API_KEY"] = args.api_key.strip()
    if API_KEY:
        print("using OpenAlex Premium key (rate limit tied to key, not IP)")
    sleep = 2.5 if args.slow else (0.3 if args.fast else 1.0)

    models, years = MODELS, YEARS
    if args.test:
        models = [("GPT-4", "OpenAI", 2023), ("Claude 3", "Anthropic", 2024),
                  ("Gemini 1.5", "Google", 2024), ("Llama 3", "Meta", 2024),
                  ("Qwen2.5", "Alibaba", 2024), ("DeepSeek-V3", "DeepSeek", 2024)]
        years = [2024, 2025]

    print(f"pace: 1 request / {sleep}s")
    rows = fetch(models, years, refresh=args.refresh, sleep=sleep)
    prov = aggregate(rows)
    providers, totals = write_shares(prov, years)

    print("\n=== provider mention counts by year ===")
    hdr = "provider".ljust(16) + "".join(str(y).rjust(9) for y in years)
    print(hdr); print("-" * len(hdr))
    for p in providers:
        print(p.ljust(16) + "".join(str(prov[p].get(y, 0)).rjust(9) for y in years))
    print("total".ljust(16) + "".join(str(totals[y]).rjust(9) for y in years))
    ly = max(years)
    if totals[ly]:
        print(f"\n=== share in {ly} ===")
        for p in providers:
            print(f"  {p.ljust(16)} {100*prov[p].get(ly,0)/totals[ly]:5.1f}%")
    if not args.test:
        plot(prov, providers, years)
        try:
            import make_report
            make_report.build()                      # writes report.html
        except Exception as e:                        # noqa: BLE001
            print(f"(report.html generation skipped: {e})")
    print(f"\nraw -> {RAW_XLSX}\nshares -> {SHARE_XLSX}")


if __name__ == "__main__":
    main()
