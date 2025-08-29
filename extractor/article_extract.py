from __future__ import annotations

import re
import unicodedata
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag, NavigableString

from extractor.cleaner import clean_html


_TR_STOPS = {
    "ve","ile","de","da","ki","bu","şu","o","bir","daha","çok","az","mi","mu","mı","mü",
    "ya","ya da","hem","gibi","ise","için","ama","fakat"
}

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36")
HDRS = {"User-Agent": UA, "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8"}

def _http_get(u: str, timeout: int = 12) -> str | None:
    try:
        r = requests.get(u, headers=HDRS, timeout=timeout, allow_redirects=True)
        if r.ok and r.text and len(r.text) > 400:
            return r.text
    except Exception:
        pass
    return None

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("İ", "i").replace("I", "ı")
    return s.lower()

def _tokens(s: str) -> list[str]:
    s = _norm(s)
    s = re.sub(r"[^a-zçğıöşü0-9\s]+", " ", s)
    return [t for t in s.split() if t and t not in _TR_STOPS]

def _title_keywords(title: str) -> set[str]:
    return set(_tokens(title)[:8])

def _title_score(text: str, title: str) -> float:
    if not text or not title:
        return 0.0
    T = _title_keywords(title)
    if not T:
        return 0.0
    head = _norm(text[:2000])
    hits = sum(1 for t in T if t in head)
    return hits / max(1, len(T))

RELATED_PAT = re.compile(
    r"(ilginizi\s*çekebilir|ilginizi\s*cekebilir|ilgini\s*çekebilir|ilgini\s*cekebilir|"
    r"ilgili|benzer|öneri|önerilen|öne çıkan|gündem|son dakika|editör|"
    r"recommended|related|trending|most read|most viewed|popular|"
    r"sıradaki|sıradaki haber|daha fazla|çok okunan|çok konuşulan|manşet)",
    re.I,
)

BAD_LINE_PAT = re.compile(
    r"(This is a modal window|Beginning of dialog window|Escape will cancel|End of dialog window|"
    r"^ilginizi\s*çekebilir:?$|^ilgili:?$|"
    r"^haber(?:in|)\s*devam[ıi]$|^içeriğin\s*devam[ıi]\s*aşağ[ıi]da$|"
    r"^s[ıi]radak[ıi]\s*haber$|^en\s*çok\s*okunan(lar|)$|^en\s*çok\s*i(?:z|)lenen(ler|)$|"
    r"^keşfet$|^popüler(\s*haberler|)$)",
    re.I,
)

PROMO_HEAD_PAT = re.compile(
    r"^\s*(sıradaki\s*haber|son\s*dakika|en\s*çok\s*okunan(lar|)|en\s*çok\s*i(?:z|)lenen(ler|)|"
    r"popüler(\s*haberler|)|öne\s*çıkan(lar|)|manşet(ler|)|keşfet|trend(ing)?|"
    r"haber(?:in|)\s*devam[ıi]|içeriğin\s*devam[ıi]\s*aşağ[ıi]da)\s*$",
    re.I
)

LIST_PAT = re.compile(r"(list|grid|cards?|manset|feed|stream|timeline)", re.I)

def _txtlen(node: Tag) -> int:
    return len(node.get_text(" ", strip=True) or "")

def _link_density(node: Tag) -> float:
    text_len = _txtlen(node)
    if text_len == 0:
        return 0.0
    link_text = " ".join(a.get_text(" ", strip=True) for a in node.find_all("a"))
    return len(link_text) / max(1, text_len)

def _looks_related(node: Tag) -> bool:
    cls = " ".join((node.get("class") or []))
    idv = node.get("id") or ""
    if RELATED_PAT.search(cls) or RELATED_PAT.search(idv):
        return True
    if node.name in ("aside", "nav", "section"):
        if _link_density(node) >= 0.5:
            return True
    if _link_density(node) >= 0.7:
        return True
    if _txtlen(node) < 120 and len(node.find_all("a")) >= 3:
        return True
    return False

def _get_title(soup: BeautifulSoup) -> Optional[str]:
    for sel in ['meta[property="og:title"]', 'meta[name="title"]', "title", "h1"]:
        el = soup.select_one(sel)
        if not el:
            continue
        t = el.get("content") if el.name == "meta" else el.get_text(" ", strip=True)
        if t and len(t) > 4:
            return t.strip()
    return None

def _get_author(soup: BeautifulSoup) -> Optional[str]:
    for sel in ['meta[name="author"]', '[itemprop="author"]', '.author, .yazar, .article-author, [rel="author"]']:
        el = soup.select_one(sel)
        if not el:
            continue
        t = el.get("content") if el.name == "meta" else el.get_text(" ", strip=True)
        if t:
            return t.strip()
    return None

def _get_date(soup: BeautifulSoup) -> Optional[str]:
    for sel in ['meta[property="article:published_time"]', 'meta[name*="date" i]', "time[datetime]", "time"]:
        el = soup.select_one(sel)
        if not el:
            continue
        t = el.get("content") or el.get("datetime") or el.get_text(" ", strip=True)
        if t and len(t) >= 6:
            return t.strip()
    return None

def _jsonld_article(soup: BeautifulSoup, page_title: Optional[str]) -> Optional[dict]:
    import json
    best, best_score = None, -1.0
    for s in soup.find_all("script", type=lambda v: v and "ld+json" in v):
        raw = s.string or s.get_text(strip=True) or ""
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        objs = data if isinstance(data, list) else [data]
        for o in objs:
            if not isinstance(o, dict):
                continue
            typ = o.get("@type")
            if isinstance(typ, list):
                typ = next((t for t in typ if isinstance(t, str)), None)
            if not isinstance(typ, str):
                continue
            if typ.lower() not in {"newsarticle","article","blogposting","reportagenewsarticle"}:
                continue
            headline = (o.get("headline") or o.get("name") or "").strip()
            body = (o.get("articleBody") or "").strip()
            if not body or len(body) < 300:
                continue
            ts = _title_score(body[:2000], page_title or headline or "")
            if ts > best_score:
                author = None
                a = o.get("author")
                if isinstance(a, dict):
                    author = a.get("name")
                elif isinstance(a, list) and a and isinstance(a[0], dict):
                    author = a[0].get("name")
                elif isinstance(a, str):
                    author = a
                best = {
                    "title": headline or None,
                    "author": author,
                    "date": o.get("datePublished") or o.get("dateCreated") or None,
                    "text": body,
                    "_title_score": ts,
                }
                best_score = ts
    return best

CAND_TOKENS = (
    "article-body", "articleBody", "content", "news-detail",
    "story-body", "post-content", "read-content", "yazi", "metin", "haber"
)

def _find_container(soup: BeautifulSoup) -> Optional[Tag]:
    best_article, best_score = None, 0
    for art in soup.find_all("article"):
        cls = " ".join((art.get("class") or [])).lower()
        if LIST_PAT.search(cls):
            continue
        ps = art.find_all("p")
        plen = len(ps)
        txt_len = sum(len(p.get_text(" ", strip=True)) for p in ps)
        link_d = _link_density(art)
        score = plen * 20 + txt_len
        if plen >= 2 and link_d < 0.55 and score > best_score:
            best_article, best_score = art, score
    if best_article:
        return best_article

    for sel in ['[itemprop="articleBody"]', "main", '[role="main"]']:
        el = soup.select_one(sel)
        if el and _txtlen(el) > 200:
            return el

    for tok in CAND_TOKENS:
        el = soup.find(True, class_=re.compile(re.escape(tok), re.I))
        if el and _txtlen(el) > 200:
            return el

    best_div, best_div_score = None, 0
    for div in soup.find_all("div"):
        cls = " ".join((div.get("class") or [])).lower()
        if LIST_PAT.search(cls):
            continue
        ps = div.find_all("p")
        plen = len(ps)
        if plen < 2:
            continue
        if _link_density(div) > 0.60:
            continue
        score = plen * 14 + _txtlen(div)
        if score > best_div_score:
            best_div, best_div_score = div, score
    return best_div

def _expand_to_content_ancestor(start: Tag, max_up: int = 3) -> Tag:
    node, best = start, start
    best_score = len(start.find_all("p")) * 10 + _txtlen(start)
    up = 0
    while node.parent and isinstance(node.parent, Tag) and up < max_up:
        node = node.parent
        if _link_density(node) > 0.6:
            up += 1
            continue
        score = len(node.find_all("p")) * 10 + _txtlen(node)
        if score > best_score:
            best, best_score = node, score
        up += 1
    return best

def _is_shouty(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 4:
        return False
    upp = sum(1 for c in letters if c.isupper() or c in "ÇĞİÖŞÜ")
    return upp / max(1, len(letters)) >= 0.8

def stitch_following(soup: BeautifulSoup, start: Tag, max_siblings: int = 80, page_title: str | None = None) -> Tag:
    T = _title_keywords(page_title or "")

    def title_like_text(txt: str) -> float:
        if not T or not txt:
            return 0.0
        toks = set(_tokens(txt))
        return len(toks & T) / max(1, len(T))

    def is_new_article(node: Tag) -> bool:
        if node.name == "article":
            return True
        cls = " ".join((node.get("class") or []))
        iid = node.get("id") or ""
        if re.search(r"(list|grid|cards?|manset|feed|stream|timeline)", cls, re.I):
            return True
        if node.find(["h2","h3"]) and _link_density(node) > 0.55:
            return True
        return False

    wrap = soup.new_tag("div")
    start = _expand_to_content_ancestor(start)
    wrap.append(start)

    sib = start.next_sibling
    taken = 0
    low_sim_streak = 0
    min_chars_before_guard = 1200
    collected_len = _txtlen(start)

    while sib and taken < max_siblings:
        if isinstance(sib, NavigableString):
            sib = sib.next_sibling
            continue
        if not isinstance(sib, Tag):
            sib = sib.next_sibling
            continue

        if is_new_article(sib):
            break

        if _looks_related(sib) or _link_density(sib) >= 0.75:
            sib = sib.next_sibling
            continue

        if _txtlen(sib) < 80 or len(sib.find_all("p")) < 1:
            sib = sib.next_sibling
            continue

        if sib.name in ("h2", "h3", "h4"):
            htxt = sib.get_text(" ", strip=True)
            if PROMO_HEAD_PAT.search(htxt):
                if collected_len >= 900:
                    break
                else:
                    sib = sib.next_sibling
                    continue

        sim = title_like_text(sib.get_text(" ", strip=True)[:800])
        if collected_len >= min_chars_before_guard and sim < 0.08:
            low_sim_streak += 1
            if low_sim_streak >= 3:
                break
            sib = sib.next_sibling
            continue
        else:
            low_sim_streak = 0

        if sib.name in ("h2", "h3", "h4"):
            htxt = sib.get_text(" ", strip=True)
            if htxt and (_is_shouty(htxt) or RELATED_PAT.search(htxt)) and sim < 0.12:
                break

        wrap.append(sib)
        collected_len += _txtlen(sib)
        taken += 1
        sib = sib.next_sibling

    return wrap

def _collect_article_text_harder(article: Tag, page_title: Optional[str] = None) -> str:
    allowed = {"p", "h2", "h3", "blockquote", "li"}
    parts: list[str] = []
    seen = set()

    def should_skip_block(n: Tag) -> bool:
        if _looks_related(n) or _link_density(n) > 0.75:
            return True
        cls = " ".join((n.get("class") or [])).lower()
        if LIST_PAT.search(cls):
            return True
        return False

    for node in article.descendants:
        if not isinstance(node, Tag):
            continue
        if node.name in {"script", "style"}:
            continue

        par = node
        skip = False
        while isinstance(par, Tag) and par is not article:
            if should_skip_block(par):
                skip = True
                break
            par = par.parent
        if skip:
            continue

        if node.name not in allowed:
            continue

        txt = node.get_text(" ", strip=True)
        if not txt:
            continue
        if BAD_LINE_PAT.search(txt):
            continue
        if node.name == "li" and len(txt) < 20:
            continue
        if node.name in {"h2", "h3"}:
            if len(txt) < 8 or RELATED_PAT.search(txt) or _is_shouty(txt) or PROMO_HEAD_PAT.search(txt):
                continue

        if txt in seen:
            continue
        seen.add(txt)
        parts.append(txt)

    text = "\n".join(parts).strip()
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _truncate_on_promo(text, min_keep=900)
    return text

def _collect_text(n: Tag) -> str:
    parts = []
    for el in n.find_all(["h2", "h3", "p", "li", "blockquote"]):
        t = el.get_text(" ", strip=True)
        if not t:
            continue
        if BAD_LINE_PAT.search(t):
            continue
        if el.name in ("h2", "h3") and (_is_shouty(t) or RELATED_PAT.search(t) or PROMO_HEAD_PAT.search(t)):
            continue
        if el.name == "li" and len(t) < 20:
            continue
        parts.append(t)
    text = "\n".join(parts).strip()
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _truncate_on_promo(text, min_keep=900)
    return text

def _truncate_on_promo(text: str, min_keep: int = 900) -> str:
    if not text:
        return text
    lines = text.splitlines()
    acc = 0
    for i, l in enumerate(lines):
        s = l.strip()
        acc += len(s) + 1
        if PROMO_HEAD_PAT.search(s):
            if acc >= min_keep:
                return "\n".join(lines[:i]).strip()
    return text

def _boundary_cut(text: str, signals: int = 3, min_keep: int = 1200, max_cut_ratio: float = 0.25) -> str:
    if not text or len(text) < min_keep or signals >= 9999:
        return text
    original_len = len(text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    bad_words = re.compile(
        r"(ilginizi\s*çekebilir|ilgili|öneri|benzer|trending|related|çok okunan(lar|)|çok konuşulan|"
        r"son dakika|gündem|editör|manşet|sıradaki\s*haber|öne çıkan|video|galeri|canlı|izle|"
        r"haber(?:in|)\s*devam[ıi]|içeriğin\s*devam[ıi]\s*aşağ[ıi]da|"
        r"en\s*çok\s*i(?:z|)lenen(ler|)|yorumlar(\s*ve\s*emojiler)?\s*aşağ[ıi]da|"
        r"keşfet|popüler(\s*haberler|)|sana\s*özel\s*seçtiklerimiz|son\s*eklenenler)",
        re.I,
    )
    count = 0
    cut_idx = None
    for i, l in enumerate(lines):
        bad_hit = (len(l) < 160 and bad_words.search(l)) or _is_shouty(l)
        if bad_hit:
            count += 1
        else:
            count = max(0, count - 1)
        if count >= signals and i > 4:
            cut_idx = i - signals + 1
            break
    if cut_idx is not None:
        kept = "\n".join(lines[:cut_idx]).strip()
        if len(kept) >= min_keep and (original_len - len(kept)) / original_len <= max_cut_ratio:
            return kept
    return text

def _drop_trailing_unrelated(text: str, title: str) -> str:
    if not text or not title or len(text) < 900:
        return text
    T = _title_keywords(title)
    sents = re.split(r'(?<=[\.\!\?])\s+', text)

    def sent_score(s: str) -> float:
        if not T:
            return 0.0
        toks = set(_tokens(s))
        return len(toks & T) / max(1, len(T))

    streak = 0
    cut = None
    for i, s in enumerate(sents):
        shortish = len(s) < 170
        if shortish and sent_score(s) < 0.08:
            streak += 1
        else:
            streak = 0
        if streak >= 7 and i > 6:
            cut = i - streak + 1
            break
    if cut:
        trimmed = " ".join(sents[:cut]).strip()
        if len(trimmed) >= max(700, int(len(text) * 0.65)):
            return trimmed
    return text

def _find_next_url(soup: BeautifulSoup, base_url: str) -> str | None:
    el = soup.select_one('link[rel="next"], a[rel="next"]')
    if el and el.get("href"):
        return urljoin(base_url, el["href"])
    for a in soup.find_all("a", href=True):
        txt = (a.get_text(" ", strip=True) or "").lower()
        if txt in ("sonraki", "devam", "next", "ileri") or "next" in " ".join(a.get("class") or []):
            return urljoin(base_url, a["href"])
    return None

def _collect_from_soup(soup: BeautifulSoup, title: str | None, max_siblings: int) -> str:
    cont0 = _find_container(soup)
    text = ""
    if cont0:
        if isinstance(cont0, Tag) and cont0.name == "article":
            text = _collect_article_text_harder(cont0, title)
            para_count = len(cont0.find_all("p"))
            if len(text) < 600 or para_count < 3:
                cont = stitch_following(soup, cont0, max_siblings=max_siblings, page_title=title)
                t2 = _collect_text(cont)
                if len(t2) > len(text):
                    text = t2
        else:
            cont = stitch_following(soup, cont0, max_siblings=max_siblings, page_title=title)
            text = _collect_text(cont)
    return text

def extract_article(
    html: str,
    url: str,
    min_len: int = 700,
    boundary_signals: int = 3,
    boundary_min_keep: int = 900,
    max_siblings: int = 80,
    multipage: int = 3,
    no_boundary: bool = False,
) -> dict:
    cleaned_html = clean_html(html)
    soup = BeautifulSoup(cleaned_html, "lxml")

    title = _get_title(soup)
    author = _get_author(soup)
    date = _get_date(soup)

    text = _collect_from_soup(soup, title, max_siblings=max_siblings)

    if len(text) < max(350, int(min_len * 0.6)):
        try:
            from readability import Document
            doc = Document(html)
            summary = doc.summary(html_partial=True)
            rsoup = BeautifulSoup(clean_html(summary), "lxml")
            rtext = _collect_from_soup(rsoup, title, max_siblings=max_siblings)
            if len(rtext) > len(text):
                text = rtext
        except Exception:
            pass

    cand_b = cand_c = ""
    try:
        from trafilatura import extract as t_extract
        cand_b = (t_extract(
            html, url=url, with_metadata=False, include_comments=False,
            favor_recall=False, target_language="tr"
        ) or "").strip()
        cand_c = (t_extract(
            html, url=url, with_metadata=False, include_comments=False,
            favor_recall=True, target_language="tr"
        ) or "").strip()
    except Exception:
        pass

    jsonld = _jsonld_article(soup, title)
    cand_d = (jsonld.get("text") if jsonld else "") or ""
    cand_a = (text or "").strip()

    def score_tuple(txt: str) -> tuple[float, int]:
        return (_title_score(txt, title or ""), len(txt or ""))

    cands = []
    for label, txt in (("container", cand_a), ("traf_prec", cand_b), ("traf_recall", cand_c), ("jsonld", cand_d)):
        if txt and len(txt) >= 200:
            ts, ln = score_tuple(txt)
            cands.append((label, txt, ts, ln))

    if not cands:
        chosen = "container"
        text = cand_a
    else:
        best = max(
            cands,
            key=lambda x: (x[2] >= 0.12, x[3] >= min_len, min(x[3], 6000), round(x[2], 3))
        )
        chosen = best[0]
        text = best[1]
        if title and _title_score(text, title) < 0.08:
            safer = next((t for (lbl, t, ts, ln) in cands if lbl == "container" and ts >= 0.06), None)
            if safer and len(safer) >= min(600, int(len(text)*0.8)):
                chosen = "container"
                text = safer

    if jsonld:
        title = title or jsonld.get("title") or title
        author = author or jsonld.get("author") or author
        date = date or jsonld.get("date") or date

    if multipage and len(text) < min_len * 3:
        seen = {url}
        cur_url, cur_soup = url, soup
        added = []
        for _ in range(max(1, multipage)):
            nxt = _find_next_url(cur_soup, cur_url)
            if not nxt or nxt in seen:
                break
            h2 = _http_get(nxt, timeout=12)
            if not h2:
                break
            s2 = BeautifulSoup(clean_html(h2), "lxml")
            t2 = _collect_from_soup(s2, title, max_siblings=max_siblings)
            if t2 and len(t2) > 100:
                if title and _title_score(t2[:1500], title) < 0.06:
                    pass
                else:
                    added.append(t2)
            seen.add(nxt)
            cur_url, cur_soup = nxt, s2
        if added:
            text = (text + "\n\n" + "\n\n".join(added)).strip()

    text_before = text
    if not no_boundary:
        dyn_min_keep = max(1200, int(len(text_before) * 0.55))
        text = _drop_trailing_unrelated(text, title or "")
        text = _boundary_cut(text, signals=max(3, boundary_signals), min_keep=dyn_min_keep, max_cut_ratio=0.25)
        if len(text) < max(700, int(len(text_before) * 0.65)):
            text = text_before

    if len(text) < min_len:
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 15]
        blob = "\n".join(paras)
        if len(blob) > len(text) and _title_score(blob, title or "") >= _title_score(text, title or "") - 0.05:
            text = blob

    text = re.sub(r"\n{3,}", "\n\n", (text or "")).strip()
    return {"title": title, "author": author, "date": date, "text": text}
