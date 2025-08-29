from __future__ import annotations
import re
from urllib.parse import urljoin, urlparse, urlunparse
from typing import Optional, List
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
import os


UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36")
HDRS = {"User-Agent": UA, "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8"}

def _strip(u: str) -> str:
    u2 = re.sub(r"[?&](utm_[^=&]+|gclid|fbclid|ocid|mcid|ref)=[^&]*", "", u, flags=re.I)
    return u2.rstrip("?&")

def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html or "", "lxml")

def _get(url: str, timeout: int = 12) -> Optional[str]:
    try:
        r = requests.get(_strip(url), headers=HDRS, timeout=timeout, allow_redirects=True)
        if r.ok and r.text and len(r.text) > 400:
            return r.text
    except Exception:
        pass
    return None

def _find_rel(html: str, base: str, rel: str) -> Optional[str]:
    try:
        el = _soup(html).select_one(f'link[rel*="{rel}" i]')
        if el and el.get("href"):
            target = urljoin(base, el["href"])
            return target
    except Exception:
        pass
    return None

def _find_amp(html: str, base: str) -> Optional[str]:
    return _find_rel(html, base, "amphtml")

def _find_print(html: str, base: str) -> Optional[str]:
    s = _soup(html)
    cand = s.select_one('link[rel="alternate"][media*="print" i]') or s.find("a", href=re.compile("print", re.I))
    href = urljoin(base, cand["href"]) if cand and cand.get("href") else None
    return href

def _find_canonical(html: str, base: str) -> Optional[str]:
    try:
        href = _soup(html).select_one('link[rel="canonical"]')
        if href and href.get("href"):
            cand = urljoin(base, href["href"])
            if urlparse(cand).hostname and urlparse(cand).hostname != urlparse(base).hostname:
                return cand
    except Exception:
        pass
    return None

def _variants(url: str) -> List[str]:
    u = urlparse(_strip(url))._replace(fragment="")
    base = urlunparse(u)
    path = u.path.rstrip("/")
    out = []
    for q in ("amp=1","amp=true","output=amp","print=1","output=print"):
        sep = "&" if u.query else "?"
        out.append(base + f"{sep}{q}")
    out.append(urlunparse(u._replace(path=path + "/amp")))
    out.append(urlunparse(u._replace(path="/amp" + ("/" + path.lstrip("/")))))
    uniq, seen = [], set()
    for v in out:
        if v not in seen and v != url:
            seen.add(v); uniq.append(v)
    return uniq

def _render_with_playwright(url: str, timeout_ms: int) -> str:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        try:
            ctx = b.new_context(user_agent=UA, viewport={"width":1280,"height":1800})
            page = ctx.new_page()
            page.set_default_timeout(timeout_ms)
            page.set_default_navigation_timeout(timeout_ms)

            def _goto_with_fallback(u: str) -> None:
                try:
                    page.goto(u, wait_until="networkidle")
                except PWTimeout:
                    page.goto(u, wait_until="domcontentloaded")
                page.wait_for_timeout(1200)

            _goto_with_fallback(url)

            js_rm = """
            (() => {
              const sels = [
                '[id*="cookie" i]','[class*="cookie" i]',
                '[id*="consent" i]','[class*="consent" i]',
                '[class*="gdpr" i]','[class*="overlay" i]',
                '[class*="popup" i]','[role="dialog"]'
              ];
              for (const s of sels) document.querySelectorAll(s).forEach(e=>{try{e.remove()}catch(_){}})
            })();
            """
            page.evaluate(js_rm)
            for txt in ["Kabul","Kabul et","Tamam","Anladım","Accept","I agree"]:
                try: page.get_by_text(txt, exact=False).first.click(timeout=700)
                except: pass

            for txt in ["Devamını oku","Daha fazla","Tümünü oku","Tümünü göster","Read more","Show more","Continue reading"]:
                try: page.get_by_text(txt, exact=False).first.click(timeout=800)
                except: pass
            for _ in range(8):
                page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                page.wait_for_timeout(450)

            html = page.content()

            amp = _find_amp(html, url)
            if amp:
                _goto_with_fallback(amp); return page.content()
            prn = _find_print(html, url)
            if prn:
                _goto_with_fallback(prn); return page.content()

            if len(html) < 1000:
                mctx = b.new_context(
                    user_agent=("Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
                                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"),
                    viewport={"width":390,"height":844}, is_mobile=True, has_touch=True)
                mp = mctx.new_page(); mp.set_default_timeout(timeout_ms)
                mp.set_default_navigation_timeout(timeout_ms)
                try:
                    mp.goto(url, wait_until="networkidle")
                except PWTimeout:
                    mp.goto(url, wait_until="domcontentloaded")
                for _ in range(6):
                    mp.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                    mp.wait_for_timeout(350)
                return mp.content()

            return html
        finally:
            b.close()

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.8, min=0.5, max=2))
def fetch_html(url: str, timeout: int = 12, strategy: str = "auto") -> str:
    url = _strip(url)
    if strategy == "browser":
        return _render_with_playwright(url, timeout_ms=int(float(os.getenv("PLAYWRIGHT_TIMEOUT", "30000"))))

    html = _get(url, timeout=timeout)
    if html:
        canon = _find_canonical(html, url)
        if canon:
            c_html = _get(canon, timeout=timeout)
            if c_html and len(c_html) > 400: 
                html, url = c_html, canon
        amp = _find_amp(html, url)
        if amp:
            a = _get(amp, timeout=timeout)
            if a and len(a) > 400: 
                return a
        prn = _find_print(html, url)
        if prn:
            p = _get(prn, timeout=timeout)
            if p and len(p) > 400: 
                return p
        for v in _variants(url):
            alt = _get(v, timeout=timeout)
            if alt and len(alt) > 400:
                return alt
        return html

    for v in _variants(url):
        alt = _get(v, timeout=timeout)
        if alt:
            return alt

    return _render_with_playwright(url, timeout_ms=int(float(os.getenv("PLAYWRIGHT_TIMEOUT", "30000"))))
