from __future__ import annotations
import os, re, json, time, unicodedata, string, hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

# Google GenAI client 
try:
    from google import genai
    from google.genai.types import GenerateContentConfig
    from google.genai import errors as genai_errors
except Exception:
    genai = None
    GenerateContentConfig = None
    genai_errors = None

# prompt builder
from prompts.tweet_prompts import build_facts_prompt, build_summary_and_multi_tweet_prompt

# --- Robust JSON helpers ------------------------------------------------------
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.S)

def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = _JSON_FENCE_RE.sub("", s)
    return s.strip()

def _extract_first_json_object(s: str) -> str:
    start = s.find("{")
    if start == -1:
        return s
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return s

def _remove_trailing_commas(s: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", s)

def _json_loads_relaxed(s: str) -> Any:
    s1 = _strip_code_fences(s or "")
    try:
        return json.loads(s1)
    except Exception:
        s2 = _extract_first_json_object(s1)
        try:
            return json.loads(s2)
        except Exception:
            s3 = _remove_trailing_commas(s2)
            return json.loads(s3)


# Limits / defaults

_MAX_INPUT_CHARS = int(os.getenv("GENAI_MAX_INPUT_CHARS", "24000"))
_DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-2.0-flash-lite,gemini-1.5-flash-8b,gemini-1.5-flash",
    ).split(",")
    if m.strip()
]

# Central tweet limits 
SINGLE_TWEET_LIMIT = int(os.getenv("TWEET_SINGLE_LIMIT", "340"))   
THREAD_TWEET_LIMIT = int(os.getenv("TWEET_THREAD_LIMIT", "360"))   
SOFT_CAP = os.getenv("SOFT_CAP", "1") == "1"                       #  don't hard-cut

# Simple disk cache
_CACHE_DIR = Path(os.getenv("CACHE_DIR", ".tweet_cache"))
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Helpers

def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _slice_safe(s: str, limit: int) -> str:
    s = s or ""
    return s if len(s) <= limit else s[:limit]

def _get_client_and_model(model: Optional[str] = None) -> Tuple[Any, str]:
    if genai is None or GenerateContentConfig is None:
        raise RuntimeError("google-genai package not found. pip install google-genai")
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_AI_STUDIO_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY missing.")
    client = genai.Client(api_key=api_key)
    model_name = model or _DEFAULT_MODEL
    if not model_name:
        raise RuntimeError("Model name is empty. Set GEMINI_MODEL.")
    return client, model_name

def _is_retryable(e: Exception) -> bool:
    if genai_errors and isinstance(e, genai_errors.ServerError):
        code = getattr(e, "status_code", None)
        if code in (429, 503):
            return True
    msg = str(e).lower()
    if "429" in msg or "quota" in msg or "rate limit" in msg:
        return True
    return False

def _generate_with_failover(
    *,
    client: Any,
    primary_model: str,
    contents: str,
    config: GenerateContentConfig,
    expect_json: bool = False,
    attempts_per_model: int = 2,
    backoff_seconds: List[float] = (1.5, 3.0, 6.0),
) -> Tuple[str, str]:
    models = [primary_model] + [m for m in _FALLBACK_MODELS if m and m != primary_model]
    last_err: Optional[Exception] = None

    for model_name in models:
        for attempt in range(attempts_per_model):
            try:
                resp = client.models.generate_content(
                    model=model_name, contents=contents, config=config
                )
                text = (getattr(resp, "text", None) or "").strip()
                if expect_json and (not text or (not (text.startswith("{") or text.startswith("[") or text.startswith("```")))):
                    raise RuntimeError(f"Expected JSON, got: {text[:160]!r}")
                if not text:
                    raise RuntimeError("Empty response text.")
                return text, model_name
            except Exception as e:
                last_err = e
                if _is_retryable(e) and attempt < attempts_per_model - 1:
                    time.sleep(backoff_seconds[min(attempt, len(backoff_seconds) - 1)])
                    continue
                if _is_retryable(e):
                    break
                raise
    if last_err:
        raise last_err
    raise RuntimeError("Generation failed with no specific error.")


# Local fact derivation & dynamic min-facts

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")

def _sentences(s: str) -> list[str]:
    return [x.strip() for x in _SENT_SPLIT_RE.split(s or "") if x.strip()]

def _score_sentence_for_fact(s: str) -> int:
    score = 0
    if re.search(r"\d", s): score += 2
    if re.search(r"\b(ay|gün|yıl|hafta|saat|dakika)\b", s, re.I): score += 1
    if re.search(r"\b(MR|MRI|gri madde|korteks|PCC|deney|çalışma|kontrol|tutuklandı|açıklandı|karar|soruşturma|rekor|proje)\b", s, re.I): score += 2
    if re.search(r"\b(üniversite|enstitü|bakanlık|mahkeme|kurulu|TSK|Meclis|Yargıtay|Belediyesi)\b", s, re.I): score += 1
    if len(s) <= 180: score += 1
    return score

def _derive_must_facts_from_text(title: str, summary: str, text: str, max_facts: int = 8) -> list[str]:
    pool = _sentences(summary) + _sentences(text)
    if not pool:
        pool = _sentences(title)
    cand = sorted(pool, key=_score_sentence_for_fact, reverse=True)
    out, seen = [], set()
    for c in cand:
        k = re.sub(r"\W+","", c.lower())[:140]
        if k and k not in seen:
            seen.add(k); out.append(c)
        if len(out) >= max_facts: break
    if not out and (title or "").strip():
        out = [title.strip()]
    return out

def compute_min_facts(text: str) -> int:
    n_chars = len(text or "")
    n_nums  = len(re.findall(r"\d+", text or ""))
    n_caps  = len(re.findall(r"\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]{2,}\b", text or ""))
    score = 5
    if n_chars > 1500: score += 2
    if n_nums >= 3:    score += 2
    if n_caps >= 6:    score += 2
    return max(4, min(10, score))

def _rewrite_length_safe(client, model_name, text: str, limit: int, output_language: str = "Turkish") -> str:
    """Shortens the text to fit within the limit, ending with a complete sentence (LLM-based, without abrupt cuts)."""
    prompt = (
        f"Shorten the following text in {output_language}, with a maximum length of {limit} characters."
        f"Ensure it ends with a complete sentence. Do not use emojis or hashtags.\n\nTEXT:\n{text}"
    )
    cfg = GenerateContentConfig(temperature=0.2, max_output_tokens=180)
    new_text, _ = _generate_with_failover(
        client=client, primary_model=model_name, contents=prompt, config=cfg
    )
    return (new_text or "").strip()


def extract_summary_key_points(summary: str) -> List[str]:
    sentences = _sentences(summary)
    return [s for s in sentences[:5] if len(s) > 20]


# Cleanup helpers (sentence/word safe, tidy ending)

_SOURCE_TRAIL_RE = re.compile(r"(?ix)(?:^|\s)(?:via|kaynak|source)\s*:?[\s\w@#.\-]+$")

def strip_trailing_source(s: str) -> str:
    return _SOURCE_TRAIL_RE.sub("", s or "").strip()

_BREAK_RE = re.compile(r"[ \t\n\r.,;:!?…—–-]")

_END_CONJ_RE = re.compile(r"(?:\s+(?:ve|veya|ama|fakat|ancak|çünkü|lakin|yalnız))+$", re.I)

def _tidy_end(s: str, limit: int) -> str:
    if not s:
        return s
    s = s.rstrip(" ,;:—–-…")
    s = _END_CONJ_RE.sub("", s).rstrip(" ,;:—–-…")
    if not s:
        return s
    if s[-1] not in ".?!":
        if len(s) < limit:
            s = s + "."
        else:
            s = (s[:-1] + ".") if len(s) > 0 else "."
    return s

def _smart_cap(s: str, limit: int, prefer_sentence: bool = True) -> str:
    s = (s or "").strip()
    if len(s) <= limit:
        return _tidy_end(s, limit)

    
    if s.endswith(("…", "...")):
        lp = max(s.rfind("."), s.rfind("?"), s.rfind("!"))
        if lp != -1 and len(s) - lp <= 80:
            s = s[:lp + 1].rstrip()
            return _tidy_end(s, limit)

    
    if prefer_sentence:
        punct_positions = [s.rfind(p, 0, limit + 1) for p in (".", "?", "!")]
        m = max(punct_positions)
        if m != -1 and (limit - m) <= 60 and (m + 1) >= int(limit * 0.65):
            return _tidy_end(s[:m + 1].rstrip(), limit)

    
    cut = -1
    for m in _BREAK_RE.finditer(s[:limit + 1]):
        cut = m.start()
    if cut == -1 or cut < max(0, limit - 50):
        cut = limit

    out = s[:cut].rstrip()
    return _tidy_end(out, limit)

def hard_cap_280(s: str) -> str:
    """Single tweet closer: does not truncate when SOFT_CAP is enabled; otherwise trims according to SINGLE_TWEET_LIMIT."""
    s = strip_trailing_source(s)
    if SOFT_CAP:
        return s.strip()
    return _smart_cap(s, SINGLE_TWEET_LIMIT, prefer_sentence=True)

def flexible_cap_320(s: str) -> str:
    """Thread tweet closer: does not truncate when SOFT_CAP is enabled; otherwise trims according to THREAD_TWEET_LIMIT."""
    s = strip_trailing_source(s)
    if SOFT_CAP:
        return s.strip()
    return _smart_cap(s, THREAD_TWEET_LIMIT, prefer_sentence=True)


# Coverage utilities

def _normalize_text(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    allowed = set(string.ascii_lowercase + "0123456789" + "ğüşöçı ")
    s = "".join(ch if ch in allowed else " " for ch in s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _coverage_count_smart(tweet_text: str, must_facts: List[str]) -> int:
    if not must_facts:
        return 0
    tweet_lower = (tweet_text or "").lower()
    tweet_nums  = re.findall(r"\d+(?:[.,]\d+)?", tweet_lower)
    covered = 0
    for fact in must_facts:
        if not isinstance(fact, str) or not fact.strip():
            continue
        f_lower = fact.lower()
        fact_words = [w for w in re.findall(r"[a-zğüşöçı]+", f_lower) if len(w) >= 3]
        word_hits = sum(1 for w in fact_words if w in tweet_lower)
        fact_nums = re.findall(r"\d+(?:[.,]\d+)?", f_lower)
        num_hit   = any(n.replace(",",".") in [t.replace(",",".") for t in tweet_nums] for n in fact_nums)
        if num_hit or word_hits >= 2 or (len(fact_words) <= 2 and word_hits >= 1):
            covered += 1
    return covered

def _coverage_count_chain(tweet_chain: List[str], must_facts: List[str]) -> int:
    if not must_facts or not tweet_chain:
        return 0
    combined_text = " ".join(tweet_chain).lower()
    tweet_nums = re.findall(r"\d+(?:[.,]\d+)?", combined_text)
    covered = 0
    for fact in must_facts:
        if not isinstance(fact, str) or not fact.strip():
            continue
        f_lower = fact.lower()
        fact_words = [w for w in re.findall(r"[a-zğüşöçı]+", f_lower) if len(w) >= 3]
        word_hits = sum(1 for w in fact_words if w in combined_text)
        fact_nums = re.findall(r"\d+(?:[.,]\d+)?", f_lower)
        num_hit = any(n.replace(",",".") in [t.replace(",",".") for t in tweet_nums] for n in fact_nums)
        if num_hit or word_hits >= 2 or (len(fact_words) <= 2 and word_hits >= 1):
            covered += 1
    return covered


# One-tweet rewrite (ALL facts) 

def _rewrite_to_include_all_facts(
    *,
    client: Any,
    model_name: str,
    original_tweet: str,
    facts: List[str],
    persona_label: str,
    mode_style: str,
    output_language: str = "Turkish",
    temperature: float = 0.3,
    max_output_tokens: int = 420,
    as_chain: bool = False,
) -> str:
    facts_block = "\n".join([f"- {f}" for f in facts if isinstance(f, str) and f.strip()])

    if as_chain:
        instructions = (
            f"You are creating a Twitter thread in {output_language}. "
            f"Persona tone: {persona_label} (mode_style={mode_style}). "
            f"Return 2-3 connected tweets that together include ALL facts below. "
            f"Each tweet ≤{THREAD_TWEET_LIMIT} chars. Format as JSON array: [\"tweet1\", \"tweet2\", \"tweet3\"]\n\n"
            f"Original content basis:\n{original_tweet}\n\n"
            f"Facts to include (ALL mandatory across the thread):\n{facts_block}\n\n"
            f"Create thread now in {output_language}."
        )
    else:
        instructions = (
            f"You are rewriting a social-media tweet in {output_language}. "
            f"Persona tone: {persona_label} (mode_style={mode_style}). "
            "You MUST explicitly include ALL facts below (preserve key numbers, names, entities). "
            "Keep it concise and natural. You MUST NOT include any emojis or hashtags.\n\n"
            f"Original tweet:\n{original_tweet}\n\n"
            f"Facts to include (ALL mandatory):\n{facts_block}\n\n"
            f"Rewrite now as ONE tweet (≤{SINGLE_TWEET_LIMIT} chars) in {output_language}."
        )

    config = GenerateContentConfig(temperature=temperature, max_output_tokens=max_output_tokens)
    text_out, _ = _generate_with_failover(
        client=client, primary_model=model_name, contents=instructions, config=config,
        expect_json=as_chain
    )
    return text_out.strip()


# Cache helpers

def _cache_key(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _cache_read(key: str) -> Optional[Dict[str, Any]]:
    fp = _CACHE_DIR / f"{key}.json"
    if fp.exists():
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def _cache_write(key: str, data: Dict[str, Any]) -> None:
    fp = _CACHE_DIR / f"{key}.json"
    try:
        fp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    prune_cache(_CACHE_DIR, max_age_days=2)

def prune_cache(dir_path: Path, max_age_days: int = 7):
    """Remove cache files older than `max_age_days` days to prevent cache bloating."""
    import time
    cutoff = time.time() - max_age_days * 86400
    for p in dir_path.glob("*.json"):
        try:
            if p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
        except Exception:
            pass


# Facts extraction

def extract_key_facts(
    *,
    title: str,
    text: str,
    model: Optional[str] = None,
    max_facts: int = 8,
    output_language: str = "Turkish",
) -> Dict[str, Any]:
    client, model_name = _get_client_and_model(model)
    contents = build_facts_prompt(
        title=_collapse_ws(_slice_safe(title, 2000)),
        text=_collapse_ws(_slice_safe(text, _MAX_INPUT_CHARS)),
        max_facts=max_facts,
        output_language=output_language,
    )
    config = GenerateContentConfig(
        temperature=0.2, max_output_tokens=512, response_mime_type="application/json"
    )
    raw_json, _ = _generate_with_failover(
        client=client, primary_model=model_name, contents=contents, config=config, expect_json=True
    )
    try:
        data = _json_loads_relaxed(raw_json)
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    data.setdefault("top_facts", [])
    return data


# Main: summary + journalist + personas (JSON)

def summarize_and_multi_tweet(
    *,
    title: str,
    text: str,
    mode_style: str = "auto",
    persona_keys: List[str] | None = None,
    max_persona_tweets: int = 8,
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_output_tokens: int = 1100,
    output_language: str = "Turkish",
    min_facts_to_include: Optional[int] = None,
    include_all_facts: bool = False,
    use_cache: bool = True,
    enable_tweet_chains: bool = True,
    tweets_per_persona: int = 3,
) -> Dict[str, Any]:
    """
    Returns JSON with:
      - summary
      - journalist_tweet (or journalist_chain)
      - persona_tweets (single tweets or chains)
      - coverage_report
    """
    client, model_name = _get_client_and_model(model)

    # 1) Extract key facts (only when chains enabled)
    external_facts: List[str] = []
    if enable_tweet_chains:
        try:
            facts_json = extract_key_facts(
                title=title, text=text, model=model_name, output_language=output_language
            )
            external_facts = [f for f in (facts_json.get("top_facts") or []) if isinstance(f, str)]
            external_facts = external_facts[:8]
        except Exception:
            external_facts = []

    # 2) Cache key
    ck = _cache_key({
        "title": (title or "").strip(),
        "text": (text or "")[:20000],
        "mode_style": mode_style,
        "persona_keys": persona_keys or ["*ALL*"],
        "max_persona_tweets": int(max_persona_tweets),
        "model": model or _DEFAULT_MODEL,
        "temperature": float(temperature),
        "max_output_tokens": int(max_output_tokens),
        "output_language": output_language,
        "min_facts_to_include": int(min_facts_to_include) if isinstance(min_facts_to_include, int) else "dyn",
        "include_all_facts": bool(include_all_facts),
        "external_facts": external_facts,
        "enable_tweet_chains": bool(enable_tweet_chains),
        "tweets_per_persona": int(tweets_per_persona),
    })
    if use_cache:
        cached = _cache_read(ck)
        if cached:
            return cached

    # 3) Build prompt
    dyn_min = compute_min_facts(text)
    eff_min_seed = (min_facts_to_include if isinstance(min_facts_to_include, int) else dyn_min)

    temp_summary = text[:500]
    summary_facts = _derive_must_facts_from_text(title, temp_summary, text, max_facts=4)
    all_facts = external_facts + summary_facts

    contents = build_summary_and_multi_tweet_prompt(
        title=_collapse_ws(_slice_safe(title, 2000)),
        text=_collapse_ws(_slice_safe(text, _MAX_INPUT_CHARS)),
        mode_style=mode_style,
        persona_keys=persona_keys,
        max_persona_tweets=max_persona_tweets,
        output_language=output_language,
        must_include_facts=all_facts if all_facts else None,
        min_facts_to_include=max(eff_min_seed, 6),
        enable_tweet_chains=enable_tweet_chains,
        tweets_per_persona=tweets_per_persona,
    )
    contents += (
        "\n\nHard rules for ALL tweets:\n"
        "- Do NOT include any outlet/source names, domains, or URLs.\n"
        "- No 'Kaynak:' or 'via ...'.\n"
        "- Plain tweets only; no attribution.\n"
    )

    config = GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens * 2 if enable_tweet_chains else max_output_tokens,
        response_mime_type="application/json"
    )

    raw_json, _ = _generate_with_failover(
        client=client, primary_model=model_name, contents=contents, config=config, expect_json=True
    )

    # 4) Parse & normalize
    try:
        data = _json_loads_relaxed(raw_json)
    except Exception as e:
        raise RuntimeError(f"Invalid JSON from model (after relaxed parse): {raw_json[:200]}") from e

    if not isinstance(data, dict):
        data = {}
    data.setdefault("summary", "")
    data.setdefault("journalist_tweet", "")
    data.setdefault("persona_tweets", {})
    data.setdefault("top_facts", [])

    summary = str(data.get("summary", "")).strip()

    # Journalist: chain or single — soft-cap
    if enable_tweet_chains:
        journalist_data = data.get("journalist_tweet", "")
        if isinstance(journalist_data, list):
            tmp = []
            for t in journalist_data:
                t0 = strip_trailing_source(str(t).strip())
                if len(t0) > THREAD_TWEET_LIMIT:
                    t0 = _rewrite_length_safe(client, model_name, t0, THREAD_TWEET_LIMIT, output_language=output_language)
                t0 = flexible_cap_320(t0)
                tmp.append(t0)
            journalist = tmp
        else:
            t0 = strip_trailing_source(str(journalist_data).strip())
            if len(t0) > THREAD_TWEET_LIMIT:
                t0 = _rewrite_length_safe(client, model_name, t0, THREAD_TWEET_LIMIT, output_language=output_language)
            journalist = flexible_cap_320(t0)
    else:
        t0 = strip_trailing_source(str(data.get("journalist_tweet", "")).strip())
        if len(t0) > SINGLE_TWEET_LIMIT:
            t0 = _rewrite_length_safe(client, model_name, t0, SINGLE_TWEET_LIMIT, output_language=output_language)
        journalist = hard_cap_280(t0)

    # Personas 
    pt = data.get("persona_tweets")
    if not isinstance(pt, dict):
        persona_tweets: Dict[str, Any] = {}
    else:
        persona_tweets = {}
        for k, v in pt.items():
            if not isinstance(k, str):
                continue
            if enable_tweet_chains and isinstance(v, list):
                chain_out = []
                for t in v:
                    t0 = strip_trailing_source(str(t).strip())
                    if len(t0) > THREAD_TWEET_LIMIT:
                        t0 = _rewrite_length_safe(client, model_name, t0, THREAD_TWEET_LIMIT, output_language=output_language)
                    t0 = flexible_cap_320(t0)
                    chain_out.append(t0)
                persona_tweets[k] = chain_out
            else:
                t0 = strip_trailing_source(str(v).strip())
                if len(t0) > SINGLE_TWEET_LIMIT:
                    t0 = _rewrite_length_safe(client, model_name, t0, SINGLE_TWEET_LIMIT, output_language=output_language)
                persona_tweets[k] = (flexible_cap_320(t0) if enable_tweet_chains else hard_cap_280(t0))

    model_top = [f for f in (data.get("top_facts") or []) if isinstance(f, str)]
    derived = _derive_must_facts_from_text(title, summary, text, max_facts=8)

    must_facts: List[str] = []
    seen = set()
    for f in (external_facts + model_top + derived + summary_facts):
        k = re.sub(r"\W+","", (f or "").lower())[:140]
        if k and k not in seen:
            seen.add(k); must_facts.append(f)
        if len(must_facts) >= 8:
            break

    eff_min = (min_facts_to_include if isinstance(min_facts_to_include, int)
               else max(compute_min_facts(text), 6))
    if include_all_facts and must_facts:
        eff_min = len(must_facts)

    def _cov_adaptive(s_or_list) -> int:
        if isinstance(s_or_list, list):
            return _coverage_count_chain(s_or_list, must_facts)
        else:
            return _coverage_count_smart(str(s_or_list), must_facts)

    coverage = {
        "journalist_tweet_covered": _cov_adaptive(journalist),
        "persona_covered": {k: _cov_adaptive(v) for k, v in (persona_tweets or {}).items()},
        "min_required": eff_min,
        "must_include_facts": must_facts,
        "is_chain_format": enable_tweet_chains,
    }

    result: Dict[str, Any] = {
        "summary": summary,
        "journalist_tweet": journalist,
        "persona_tweets": persona_tweets,
        "coverage_report": coverage,
    }

    # 6) Optional post-enforcement (ALL facts)
    if include_all_facts and must_facts:
        need = len(must_facts)

        # Journalist
        got = _cov_adaptive(result["journalist_tweet"])
        attempts = 0
        while got < need and attempts < 2:
            if enable_tweet_chains:
                chain_result = _rewrite_to_include_all_facts(
                    client=client,
                    model_name=model_name,
                    original_tweet=str(result["journalist_tweet"]),
                    facts=must_facts,
                    persona_label="Journalist",
                    mode_style="serious" if mode_style == "auto" else mode_style,
                    output_language=output_language,
                    temperature=min(temperature, 0.4),
                    max_output_tokens=600,
                    as_chain=True,
                )
                try:
                    chain_data = _json_loads_relaxed(chain_result)
                    if isinstance(chain_data, list):
                        result["journalist_tweet"] = [flexible_cap_320(strip_trailing_source(str(t))) for t in chain_data]
                    else:
                        result["journalist_tweet"] = flexible_cap_320(strip_trailing_source(str(chain_result)))
                except Exception:
                    result["journalist_tweet"] = flexible_cap_320(strip_trailing_source(str(chain_result)))
            else:
                new_single = _rewrite_to_include_all_facts(
                    client=client,
                    model_name=model_name,
                    original_tweet=str(result["journalist_tweet"]),
                    facts=must_facts,
                    persona_label="Journalist",
                    mode_style="serious" if mode_style == "auto" else mode_style,
                    output_language=output_language,
                    temperature=min(temperature, 0.4),
                    max_output_tokens=360,
                    as_chain=False,
                )
                result["journalist_tweet"] = hard_cap_280(strip_trailing_source(new_single))
            got = _cov_adaptive(result["journalist_tweet"])
            attempts += 1

        # Personas
        repaired: Dict[str, Any] = {}
        for pk, tw in (result["persona_tweets"] or {}).items():
            got = _cov_adaptive(tw)
            attempts = 0
            temp_tw = tw
            while got < need and attempts < 2:
                if enable_tweet_chains and isinstance(temp_tw, list):
                    chain_result = _rewrite_to_include_all_facts(
                        client=client,
                        model_name=model_name,
                        original_tweet=" ".join(temp_tw),
                        facts=must_facts,
                        persona_label=pk,
                        mode_style=mode_style,
                        output_language=output_language,
                        temperature=min(temperature, 0.4),
                        max_output_tokens=720,
                        as_chain=True,
                    )
                    try:
                        chain_data = _json_loads_relaxed(chain_result)
                        if isinstance(chain_data, list):
                            temp_tw = [flexible_cap_320(strip_trailing_source(str(t))) for t in chain_data]
                        else:
                            temp_tw = [flexible_cap_320(strip_trailing_source(str(chain_result)))]
                    except Exception:
                        temp_tw = [flexible_cap_320(strip_trailing_source(str(chain_result)))]
                else:
                    new_single = _rewrite_to_include_all_facts(
                        client=client,
                        model_name=model_name,
                        original_tweet=str(temp_tw),
                        facts=must_facts,
                        persona_label=pk,
                        mode_style=mode_style,
                        output_language=output_language,
                        temperature=min(temperature, 0.4),
                        max_output_tokens=360,
                        as_chain=False,
                    )
                    temp_tw = hard_cap_280(strip_trailing_source(new_single))
                got = _cov_adaptive(temp_tw)
                attempts += 1
            repaired[pk] = temp_tw
        result["persona_tweets"] = repaired

    # Cache
    if use_cache:
        try:
            _cache_write(ck, result)
        except Exception:
            pass

    return result


# Batch repair helper 

def batch_repair_to_include_all_facts(
    *,
    generated: Dict[str, Any],
    must_include_facts: List[str],
    output_language: str = "Turkish",
    model: Optional[str] = None,
    enable_tweet_chains: bool = True,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """
    Given an already generated dict, try to repair journalist & persona tweets
    so that ALL facts are included (best-effort, 2 attempts per item).
    """
    client, model_name = _get_client_and_model(model)

    result = dict(generated or {})
    journalist = result.get("journalist_tweet")
    persona_tweets = dict(result.get("persona_tweets") or {})
    coverage = dict(result.get("coverage_report") or {})
    is_chain = bool(coverage.get("is_chain_format", enable_tweet_chains))

    def _cov_adaptive(s_or_list) -> int:
        if isinstance(s_or_list, list):
            return _coverage_count_chain(s_or_list, must_facts)
        else:
            return _coverage_count_smart(str(s_or_list), must_facts)

    must_facts = must_include_facts or []
    need = len(must_facts)

    # Journalist
    if journalist and need:
        got = _cov_adaptive(journalist)
        attempts = 0
        while got < need and attempts < 2:
            if is_chain and isinstance(journalist, list):
                chain_result = _rewrite_to_include_all_facts(
                    client=client,
                    model_name=model_name,
                    original_tweet=" ".join(journalist),
                    facts=must_facts,
                    persona_label="Journalist",
                    mode_style="serious",
                    output_language=output_language,
                    temperature=min(temperature, 0.4),
                    max_output_tokens=700,
                    as_chain=True,
                )
                try:
                    chain_data = _json_loads_relaxed(chain_result)
                    journalist = [flexible_cap_320(strip_trailing_source(str(t))) for t in (chain_data if isinstance(chain_data, list) else [chain_result])]
                except Exception:
                    journalist = [flexible_cap_320(strip_trailing_source(str(chain_result)))]
            else:
                new_single = _rewrite_to_include_all_facts(
                    client=client,
                    model_name=model_name,
                    original_tweet=str(journalist),
                    facts=must_facts,
                    persona_label="Journalist",
                    mode_style="serious",
                    output_language=output_language,
                    temperature=min(temperature, 0.4),
                    max_output_tokens=380,
                    as_chain=False,
                )
                journalist = hard_cap_280(strip_trailing_source(new_single))
            got = _cov_adaptive(journalist)
            attempts += 1
        result["journalist_tweet"] = journalist

    # Personas
    repaired: Dict[str, Any] = {}
    for pk, tw in persona_tweets.items():
        temp_tw = tw
        got = _cov_adaptive(temp_tw)
        attempts = 0
        while need and got < need and attempts < 2:
            if is_chain and isinstance(temp_tw, list):
                chain_result = _rewrite_to_include_all_facts(
                    client=client,
                    model_name=model_name,
                    original_tweet=" ".join(temp_tw),
                    facts=must_facts,
                    persona_label=pk,
                    mode_style="auto",
                    output_language=output_language,
                    temperature=min(temperature, 0.4),
                    max_output_tokens=720,
                    as_chain=True,
                )
                try:
                    chain_data = _json_loads_relaxed(chain_result)
                    temp_tw = [flexible_cap_320(strip_trailing_source(str(t))) for t in (chain_data if isinstance(chain_data, list) else [chain_result])]
                except Exception:
                    temp_tw = [flexible_cap_320(strip_trailing_source(str(chain_result)))]
            else:
                new_single = _rewrite_to_include_all_facts(
                    client=client,
                    model_name=model_name,
                    original_tweet=str(temp_tw),
                    facts=must_facts,
                    persona_label=pk,
                    mode_style="auto",
                    output_language=output_language,
                    temperature=min(temperature, 0.4),
                    max_output_tokens=360,
                    as_chain=False,
                )
                temp_tw = hard_cap_280(strip_trailing_source(new_single))
            got = _cov_adaptive(temp_tw)
            attempts += 1
        repaired[pk] = temp_tw
    result["persona_tweets"] = repaired

    # Update covarage 
    cov = {
        "journalist_tweet_covered": _cov_adaptive(result.get("journalist_tweet")),
        "persona_covered": {k: _cov_adaptive(v) for k, v in (result.get("persona_tweets") or {}).items()},
        "min_required": need if need else (coverage.get("min_required") or 0),
        "must_include_facts": must_facts or coverage.get("must_include_facts") or [],
        "is_chain_format": is_chain,
    }
    result["coverage_report"] = cov
    return result
