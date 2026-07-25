"""Model catalogue for measuring LLM mentions in the academic literature.

Defines what to look for:
  MODELS    (model, provider, release year) for every tracked system
  CONCEPTS  umbrella terms ("LLM", "generative AI", ...) tracked alongside the models
  weight_class()  open- vs closed-weight classification

The harvesting logic lives in fetch_papers.py, which imports this module.
"""
import os
import urllib.parse

MAILTO = "guillaume.coqueret@gmail.com"

API_KEY = os.environ.get("OPENALEX_API_KEY", "").strip()

def _auth(url):
    """Append the OpenAlex premium api_key (if API_KEY is set) to a fully-built URL."""
    if not API_KEY:
        return url
    sep = "&" if "?" in url else "?"
    return url + sep + "api_key=" + urllib.parse.quote(API_KEY)

YEARS = list(range(2020, 2027))

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

CONCEPT_PROVIDER = "(concept)"

CONCEPTS = [
    "large language model",                 # + "large language models"
    "LLM",                                  # + "LLMs", "LLM-based"
    "generative AI",                        # + "generative-AI"
    "generative artificial intelligence",
    "gen AI",                               # + "GenAI", "gen-AI"
    "foundation model",                     # + "foundation models"
]

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
