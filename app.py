from __future__ import annotations

import os
from pathlib import Path
import streamlit as st

# .env
try:
    from dotenv import load_dotenv  # type: ignore
    _ENV_LOADED = load_dotenv(Path(__file__).resolve().parent / ".env", override=True)
except Exception:
    _ENV_LOADED = False

for k in (
    "GOOGLE_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_FALLBACK_MODELS",
    "REQUEST_TIMEOUT",
    "PLAYWRIGHT_TIMEOUT",
):
    v = st.secrets.get(k, None)
    if v is not None and not os.getenv(k):
        os.environ[k] = str(v)

# Extractor
from extractor.fetcher import fetch_html
from extractor.article_extract import extract_article

# LLM — one-shot JSON + single extra batch repair
from llm.gemini_client import summarize_and_multi_tweet, batch_repair_to_include_all_facts

# ───────────────────────── Enhanced settings ─────────────────────────
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
TEMPERATURE = 0.4
MODE_STYLE = "auto"                 # "auto" | "entertainment" | "serious"
MAX_PERSONA_TWEETS = 9             
MIN_FACTS_TO_INCLUDE = None       
OUTPUT_LANGUAGE = "Turkish"
USE_CACHE = True
INCLUDE_ALL_FACTS_IN_MAIN_CALL = False
ALWAYS_DO_EXTRA_BATCH_REPAIR = True


ENABLE_TWEET_CHAINS = True          
TWEETS_PER_PERSONA = 3              

# ───────────────────────── UI ─────────────────────────
st.set_page_config(
    page_title=" Tweet Assistant",
    page_icon="🧩",
    layout="centered",
)

# Sidebar
st.sidebar.header("Tweet Format Settings")
tweet_format = st.sidebar.radio(
    "Tweet Format", 
    ["Thread (2-3 tweets)", "Single Tweet"], 
    index=0 if ENABLE_TWEET_CHAINS else 1
)
ENABLE_TWEET_CHAINS = (tweet_format == "Thread (2-3 tweets)")

if ENABLE_TWEET_CHAINS:
    TWEETS_PER_PERSONA = st.sidebar.slider("Tweets per Thread", 2, 4, 3)

detail_level = st.sidebar.select_slider(
    "Detail Level",
    ["Low", "Medium", "High", "Maximum"],
    value="High"
)

detail_mapping = {
    "Low": {"min_facts": 3, "max_tokens": 800},
    "Medium": {"min_facts": 5, "max_tokens": 1100}, 
    "High": {"min_facts": 7, "max_tokens": 1500},
    "Maximum": {"min_facts": 10, "max_tokens": 2000}
}
detail_config = detail_mapping[detail_level]

st.title("Tweet Assistant")
if ENABLE_TWEET_CHAINS:
    st.caption(f"Mode: Thread Format ({TWEETS_PER_PERSONA} tweets per persona) | Detail: {detail_level}")
else:
    st.caption(f"Mode: Single Tweet | Detail: {detail_level}")

url = st.text_input("News URL", placeholder="https://...")
go = st.button("Generate", type="primary")

if go:
    if not (url or "").strip():
        st.error("Please enter a news URL.")
        st.stop()

    # 1) Fetch & extract
    with st.spinner("Fetching page…"):
        html = fetch_html(url)
        if not html:
            st.error("Could not fetch the page or content is too short. Please check the link.")
            st.stop()

    with st.spinner("Extracting article…"):
        try:
            #url html
            art = extract_article(html=html, url=url) or {}
        except Exception as e:
            st.exception(e)
            st.stop()

    title = (art.get("title") or "").strip()
    text = (art.get("text") or "").strip()

    if not text:
        st.error("No article text could be extracted.")
        st.stop()

    # 2) Enhanced generation with configurable detail
    with st.spinner("Generating detailed summary & tweets…"):
        try:
            result = summarize_and_multi_tweet(
                title=title,
                text=text,
                mode_style=MODE_STYLE,
                persona_keys=None,                        
                max_persona_tweets=int(MAX_PERSONA_TWEETS),
                model=MODEL,
                temperature=float(TEMPERATURE),
                output_language=OUTPUT_LANGUAGE,
                min_facts_to_include=detail_config["min_facts"],
                include_all_facts=bool(INCLUDE_ALL_FACTS_IN_MAIN_CALL),
                use_cache=bool(USE_CACHE),
                enable_tweet_chains=bool(ENABLE_TWEET_CHAINS),
                tweets_per_persona=int(TWEETS_PER_PERSONA),
                max_output_tokens=int(detail_config["max_tokens"]),
            ) or {}
        except Exception as e:
            st.exception(e)
            st.stop()

    # 3) Enhanced batch repair with chain support
    if ALWAYS_DO_EXTRA_BATCH_REPAIR:
        try:
            must = (result.get("coverage_report", {}) or {}).get("must_include_facts", []) or []
            if must:
                with st.spinner("Refining tweets for maximum coverage…"):
                    result = batch_repair_to_include_all_facts(
                        generated=result,
                        must_include_facts=must,
                        output_language=OUTPUT_LANGUAGE,
                        model=MODEL,
                        enable_tweet_chains=ENABLE_TWEET_CHAINS,
                    )
        except Exception as e:
            st.info(f"Batch repair skipped due to error: {e}")

    # 4) Enhanced display with thread support
    st.subheader("📰 Özet")
    st.write(result.get("summary") or "—")

    st.subheader("🐦 Tweets")
    
    # Coverage metrics display
    coverage = result.get("coverage_report", {}) or {}
    if coverage:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Required Facts", coverage.get("min_required", 0))
        with col2:
            journalist_cov = coverage.get("journalist_tweet_covered", 0)
            st.metric("Journalist Coverage", journalist_cov)
        with col3:
            persona_cov = coverage.get("persona_covered", {}) or {}
            avg_cov = (sum(persona_cov.values()) / len(persona_cov)) if persona_cov else 0
            try:
                st.metric("Avg Persona Coverage", f"{avg_cov:.1f}")
            except Exception:
                st.metric("Avg Persona Coverage", str(avg_cov))

    # Journalist display
    jt = result.get("journalist_tweet")
    if jt:
        st.markdown("**Journalist**")
        if isinstance(jt, list) and ENABLE_TWEET_CHAINS:
            for i, tweet in enumerate(jt, 1):
                st.write(f"{i}/{len(jt)}: {tweet}")
                if i < len(jt):
                    st.write("🧵")  # Thread connector
        else:
            st.write(str(jt))

    # Personas display
    persona_tweets: dict = result.get("persona_tweets") or {}
    if persona_tweets:
        for persona, tw in persona_tweets.items():
            st.markdown(f"**{persona}**")
            if isinstance(tw, list) and ENABLE_TWEET_CHAINS:
                for i, tweet in enumerate(tw, 1):
                    st.write(f"{i}/{len(tw)}: {tweet}")
                    if i < len(tw):
                        st.write("🧵")
            else:
                st.write(str(tw))

    # Fact coverage breakdown
    with st.expander("📊 Fact Coverage Details"):
        must_facts = coverage.get("must_include_facts", [])
        if must_facts:
            st.write("**Must Include Facts:**")
            for i, fact in enumerate(must_facts, 1):
                st.write(f"{i}. {fact}")
            
            st.write("**Coverage per Persona:**")
            persona_cov = coverage.get("persona_covered", {}) or {}
            for persona, cov_count in persona_cov.items():
                percentage = (cov_count / len(must_facts) * 100) if must_facts else 0
                st.write(f"- {persona}: {cov_count}/{len(must_facts)} facts ({percentage:.0f}%)")
