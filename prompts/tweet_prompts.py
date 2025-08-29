from __future__ import annotations

__all__ = [
    "STYLES",
    "build_summary_prompt",
    "SUMMARY_PROMPT",
    "build_facts_prompt",
    "build_single_persona_prompt",
    "build_summary_and_multi_tweet_prompt",
]

def _lang_line(output_language: str = "Turkish") -> str:
    return (
        f"OUTPUT LANGUAGE: {output_language.upper()} (use native punctuation, numerals, and locale).\n"
        "- Keep month/day names and time formats natural to the locale.\n"
        "- Do NOT translate proper nouns (teams, venues, institutions)."
    )

def _constitution() -> str:
    return (
        """
KNOWLEDGE BASE: TWITTER CONSTITUTION (v2.0)

SECTION 1: CORE PRINCIPLES & EDITORIAL STANCE
- Dual Identity: Entertainment Mode (casual, witty, sarcastic) and
  Serious Mode (neutral, detached, informative).
- Priority: Engagement over traffic.
- Neutrality & Balance: No partisan stance.
- Source & Verification: Internal editorial check required, BUT
  tweets must NEVER include any outlet/source names, domains, or URLs.
- Responsible Content Management: Do not remove posts impulsively.

SECTION 2: VISUAL POLICY
- Aim to include a visual suggestion for every tweet.

SECTION 3: RED LINES
- Extra caution on violence, politics, abuse, children.

SECTION 4: SPIRIT
- Humor targets: celebrities, absurdities, etc.
- Tone: warm but not "friends".
- Use "comment gap" to invite replies.

WORKFLOW
- Default: given news text, produce one style tweet.
- Optional: if [all personae], produce tweets for all 8 personas.
"""
    ).strip()

STYLES = {
    "Provocative Commentator": "Opens debate with rhetorical questions and a deliberate comment gap.",
    "Intellectual Analyst": "Conceptual framing; connects to examples.",
    "Direct & Stern Columnist": "Short, decisive, firm judgement.",
    "Human-Centered Storyteller": "Focus on victim/actor; empathetic detail.",
    "Rights-Based Activist": "Rights violations and legal standards.",
    "Urban & Environment Reporter": "Technical, zoning, measurements.",
    "Banter Editor": "Absurdities; light sarcasm, harmless humor.",
}

# Persona detay seviyeleri (opsiyonel)
PERSONA_DETAIL_LEVELS = {
    "Intellectual Analyst": "high",
    "Human-Centered Storyteller": "high", 
    "Rights-Based Activist": "high",
    "Urban & Environment Reporter": "high",
    "Provocative Commentator": "medium",
    "Direct & Stern Columnist": "medium",
    "Banter Editor": "medium",
}

def build_summary_prompt(output_language: str = "Turkish") -> str:
    return f"""
{_lang_line(output_language)}
Summarize the news in a neutral, explanatory tone.
Preserve numbers, dates/times, key actors, official statements/decisions,
and the sequence of events.
OUTPUT: {output_language} only, about 120—180 words (1—2 short paragraphs).
Include one sentence highlighting the latest decision/status if relevant.
""".strip()

SUMMARY_PROMPT = build_summary_prompt("Turkish")

def build_facts_prompt(*, title: str, text: str, max_facts: int = 8, output_language: str = "Turkish") -> str:
    return f"""
{_constitution()}
OUTPUT LANGUAGE: {output_language.upper()} - Extract facts in {output_language}.

TASK: Extract up to {max_facts} TRUE facts as JSON.

Rules:
- Capture 5W1H in order of importance.
- Prefer concrete numbers, dates, official statements.
- Include ALL key details from summary-level information.
- No hallucinations.
- JSON only.

JSON SCHEMA:
{{
  "top_facts": ["...", "..."],
  "numbers": ["..."],
  "entities": ["..."],
  "sensitivity": "none|violence|politics|children|legal"
}}

--- NEWS TITLE ---
{title}

--- NEWS TEXT ---
{text}
""".strip()

def _render_mandatory_block(must_include_facts: list[str] | None, min_facts_to_include: int) -> str:
    if not must_include_facts:
        return ""
    items = "\n".join([f"{i+1}) {f}" for i, f in enumerate(must_include_facts)])
    return f"""
MANDATORY COVERAGE:
- Include at least {min_facts_to_include} items (or all if fewer).
- No invention.
Facts list:
{items}
""".strip()

def build_single_persona_prompt(
    *,
    title: str,
    text: str,
    persona_key: str,
    mode_style: str = "auto",
    must_include_facts: list[str] | None = None,
    min_facts_to_include: int = 3,
    output_language: str = "Turkish",
    enable_tweet_chains: bool = False,
    tweets_per_persona: int = 3,
) -> str:
    style_hint = STYLES.get(persona_key, "")
    mode_text = {
        "auto": "If content is serious, use Serious Mode; else Entertainment Mode.",
        "entertainment": "Use Entertainment Mode.",
        "serious": "Use Serious Mode.",
    }.get(mode_style, "If serious, use Serious Mode; else Entertainment.")

    must_block = _render_mandatory_block(must_include_facts, min_facts_to_include)

    if enable_tweet_chains:
        format_instruction = (
            f"Create {tweets_per_persona} connected tweets (thread format):\n"
            "- Tweet 1: Hook/main point (≤360 chars)\n"
            "- Tweet 2: Key details/context (≤360 chars)\n"
            "- Tweet 3: Analysis/conclusion (≤360 chars)\n"
            "- Each tweet standalone readable but connected\n"
            "- End each tweet with a complete sentence or a question (no trailing '...')\n"
            "- Format: Return as JSON array [\"tweet1\", \"tweet2\", \"tweet3\"]"
        )
    else:
        format_instruction = (
            "≤340 characters. No emojis/hashtags. Do NOT use emojis or hashtags under any circumstance. "
            "End with a complete sentence or a question (no trailing '...'). "
            "Format: single tweet line"
        )

    return f"""
{_constitution()}
{_lang_line(output_language)}

TASK: Write tweets in persona style.
- Persona: {persona_key} → {style_hint}
- Mode: {mode_style} → {mode_text}
- {format_instruction}
- Create a "Comment Gap" to invite replies.
- Preserve who/what/where/when/why/how.
- Do NOT include any outlet/source names, domains, or URLs.

{must_block}

--- NEWS TITLE ---
{title}

--- NEWS TEXT ---
{text}

After output: one line "[Used facts: 1,2,...]".
""".strip()

def build_summary_and_multi_tweet_prompt(
    *,
    title: str,
    text: str,
    mode_style: str = "auto",
    persona_keys: list[str] | None = None,
    max_persona_tweets: int = 4,
    output_language: str = "Turkish",
    must_include_facts: list[str] | None = None,
    min_facts_to_include: int = 6,
    enable_tweet_chains: bool = True,
    tweets_per_persona: int = 3,
) -> str:
    selected_personae = persona_keys or list(STYLES.keys())
    selected_personae = selected_personae[: max(0, int(max_persona_tweets))]
    persona_desc = {k: STYLES.get(k, "") for k in selected_personae}
    must_block = _render_mandatory_block(must_include_facts, min_facts_to_include)
    numbered_facts = "\n".join([f"{i+1}) {f}" for i, f in enumerate(must_include_facts)]) if must_include_facts else ""

    if enable_tweet_chains:
        persona_schema = ", ".join([f'"{k}": ["string", "string", "string"]' for k in selected_personae])
        coverage_schema = ", ".join([f'"{k}": [1,2,3]' for k in selected_personae])
        format_rules = f"""
TWEET CHAIN FORMAT (Per Persona = {tweets_per_persona} tweets):
- Tweet 1: Hook/main announcement (≤360 chars)
- Tweet 2: Key details/numbers/context (≤360 chars) 
- Tweet 3: Analysis/impact/conclusion (≤360 chars)
- Each tweet must be standalone readable and end with a full sentence/question (no trailing '...')
- Together they must cover summary-level detail
- Maintain persona voice throughout chain
- Absolutely NO emojis/hashtags in any tweet. Do NOT use emojis or hashtags under any circumstance.
"""
        json_schema = f"""
{{
  "summary": "string",
  "journalist_tweet": ["string", "string", "string"],
  "persona_tweets": {{
    {persona_schema}
  }},
  "top_facts": ["string", "string", "string"],
  "covered_fact_indexes": {{
    "journalist_tweet": [1,2,3],
    "persona_tweets": {{
      {coverage_schema}
    }}
  }}
}}"""
    else:
        persona_schema = ", ".join([f'"{k}": "string"' for k in selected_personae])
        coverage_schema = ", ".join([f'"{k}": [1,2,3]' for k in selected_personae])
        format_rules = (
            "SINGLE TWEET FORMAT: Each persona gets one tweet ≤340 chars. "
            "End with a complete sentence or a question (no trailing '...')."
        )
        json_schema = f"""
{{
  "summary": "string", 
  "journalist_tweet": "string",
  "persona_tweets": {{
    {persona_schema}
  }},
  "top_facts": ["string", "string", "string"],
  "covered_fact_indexes": {{
    "journalist_tweet": [1,2,3],
    "persona_tweets": {{
      {coverage_schema}
    }}
  }}
}}"""

    return f"""
{_constitution()}
{_lang_line(output_language)}

TASK: Return STRICT JSON with:
1) SUMMARY (neutral, comprehensive).
2) journalist_tweet (neutral, full context).
3) persona_tweets (per selected personae).
4) top_facts (6—8 concise items you used).

GLOBAL RULES:
- No emojis/hashtags. Do NOT use emojis or hashtags under any circumstance.
- Preserve who/what/where/when/why/how.
- ALL PARTS (summary + ALL tweets) MUST be written in {output_language.upper()} ONLY.
- SUMMARY must be ~120—180 words, covering ALL important details from news.
- Tweets must NEVER include any outlet/source names, domains, or URLs.
- journalist_tweet should be as informative as the summary; persona_tweets should keep persona voice.
- Each tweet must end with a full sentence or a question (no trailing '...').

{format_rules}

ROLE-BASED FACT COVERAGE (CRITICAL):
- journalist_tweet: include AT LEAST {min_facts_to_include} distinct facts (numbers/dates/entities).
- persona_tweets: include 3–4 key facts MAXIMUM; PRIORITIZE persona voice and tone. 
  Do not mechanically enumerate facts; keep the style distinct for each persona.

STYLE PRIORITY:
- For persona_tweets, persona voice > fact density (within the 3–4 fact target).
- Avoid repetitive wording across personas; vary syntax and framing.

{must_block}

SELECTED PERSONAE:
{persona_desc}

JSON SCHEMA:
{json_schema}

CONTENT:
--- TITLE ---
{title}

--- TEXT ---
{text}

FACTS LIST:
{numbered_facts if numbered_facts else "(No external facts provided)"} 

OUTPUT INSTRUCTIONS:
- Return ONLY JSON as per schema.
- For persona_tweets, include exactly these keys: {selected_personae}.
- For covered_fact_indexes, use 1-based indexes from the Facts list, or from your 'top_facts' if external list is absent.
- If fewer than {min_facts_to_include} facts exist, use all available for journalist_tweet.
- PRIORITY: Information density should primarily be in the journalist_tweet; personas keep distinct style with 3–4 facts.
""".strip()
