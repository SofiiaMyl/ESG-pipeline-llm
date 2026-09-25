import os
import csv
import time
import json
import re
import html
import hashlib
from datetime import datetime, date
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode, unquote

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from .driver_utils import get_driver

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36"
    )
}

NON_HTML_EXTENSIONS = (
    ".doc", ".docx", ".xls", ".xlsx", ".zip", ".rar", ".7z",
    ".ppt", ".pptx", ".rtf", ".odt", ".ods", ".csv", ".xml",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".mp4", ".avi", ".mov", ".mkv", ".mp3", ".wav", ".flac"
)

SKIP_SCHEMES = ("mailto:", "tel:", "javascript:", "#")

TRACKING_QUERY_KEYS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "yclid", "fbclid", "_openstat"
}

MAX_TEXT_CHARS = 20000
MIN_TEXT_CHARS = 120

NOISE_RE = re.compile(
    r"\barray\s*\(|=>\s*\'|~ID\'|IBLOCK_ID|SHOW_COUNTER|DATE_CREATE|ACTIVE_FROM_X|"
    r"\bvar\s+dataLayer\b|\bwindow\.__INITIAL_STATE__\b",
    re.IGNORECASE
)

BOILERPLATE_RE = re.compile(
    r"(^|[\W_])("
    r"cookie|consent|breadcrumb|subscribe|newsletter|social|share|related|recommend|"
    r"advert|banner|popup|modal|login|signin|search-form|sitemap|pagination|"
    r"site-nav|main-nav|top-nav|footer|copyright|toolbar|tags"
    r")([\W_]|$)",
    re.IGNORECASE
)

PDF_MAX_BYTES = 25 * 1024 * 1024
PDF_MAX_PER_COMPANY = 80
PDF_ALLOWED_MIME = ("application/pdf", "application/x-pdf")
PDF_EXTENSIONS = (".pdf",)

CONF_SCORE = {"none": 0, "low": 1, "medium": 2, "high": 3, "very_high": 4}
SOURCE_PRIORITY = {
    "jsonld": 60,
    "meta": 55,
    "time": 45,
    "date_hint": 35,
    "url": 30,
    "http": 25,
    "fallback": 20,
    "text": 10,
}

MONTHS = {
    "января": 1, "январь": 1, "янв": 1,
    "февраля": 2, "февраль": 2, "фев": 2,
    "марта": 3, "март": 3, "мар": 3,
    "апреля": 4, "апрель": 4, "апр": 4,
    "мая": 5, "май": 5,
    "июня": 6, "июнь": 6, "июн": 6,
    "июля": 7, "июль": 7, "июл": 7,
    "августа": 8, "август": 8, "авг": 8,
    "сентября": 9, "сентябрь": 9, "сен": 9, "сент": 9,
    "октября": 10, "октябрь": 10, "окт": 10,
    "ноября": 11, "ноябрь": 11, "ноя": 11,
    "декабря": 12, "декабрь": 12, "дек": 12,
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

DATE_HINT_RE = re.compile(
    r"(date|time|publish|posted|created|updated|modified|lastmod|"
    r"дата|опублик|обновл|публикац)",
    re.IGNORECASE
)

UPDATED_HINT_RE = re.compile(
    r"(updated|modified|lastmod|обновл|редакт)",
    re.IGNORECASE
)

PUBLISHED_KEYS = {
    "article:published_time", "article:published",
    "og:published_time", "datepublished", "publishdate", "pubdate",
    "dc.date", "dc.date.issued", "citation_publication_date",
    "datecreated", "created", "publication_date"
}

MODIFIED_KEYS = {
    "article:modified_time", "og:updated_time",
    "datemodified", "lastmod", "last-modified", "modified",
    "dateupdated", "updated"
}


def normalize_whitespace(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", html.unescape(str(text))).strip()


def normalize_url(url):
    if not url:
        return None
    url = str(url).strip()
    if not url or any(url.lower().startswith(s) for s in SKIP_SCHEMES):
        return None
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)

    query_pairs = parse_qsl(parsed.query, keep_blank_values=False)
    query_pairs = [(k, v) for k, v in query_pairs if k.lower() not in TRACKING_QUERY_KEYS]
    clean_query = urlencode(query_pairs, doseq=True)

    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")

    return urlunparse((parsed.scheme, parsed.netloc.lower(), path, "", clean_query, ""))


def _is_html_like_url(url):
    path = (urlparse(url).path or "").lower()
    return not any(path.endswith(ext) for ext in NON_HTML_EXTENSIONS)


def _looks_like_pdf_url(url: str) -> bool:
    p = (urlparse(url).path or "").lower()
    return p.endswith(PDF_EXTENSIONS) or ".pdf" in p


def _safe_filename(name: str) -> str:
    name = unquote(name or "")
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:180] if name else "file"


def _pdf_filename_from_url(url: str) -> str:
    p = urlparse(url).path
    base = os.path.basename(p)
    base = _safe_filename(base)
    if not base.lower().endswith(".pdf"):
        base = (base + ".pdf") if base else "file.pdf"
    return base


def _looks_like_noise(text: str) -> bool:
    t = normalize_whitespace(text)
    if not t:
        return True
    if NOISE_RE.search(t):
        cnt = len(re.findall(r"(IBLOCK_ID|SHOW_COUNTER|=>|\barray\s*\()", t, flags=re.IGNORECASE))
        if cnt >= 3:
            return True
    letters = sum(ch.isalpha() for ch in t)
    ratio = letters / (len(t) + 1e-9)
    if len(t) > 1500 and ratio < 0.35:
        return True
    return False


def _is_boilerplate_by_attr(tag):
    attrs_blob = " ".join([
        tag.get("id", ""),
        " ".join(tag.get("class", [])),
        tag.get("role", ""),
        tag.get("aria-label", ""),
    ])
    return bool(BOILERPLATE_RE.search(attrs_blob))


def _pick_main_content_node(soup):
    candidates = []
    for selector in (
        "article",
        "main",
        '[role="main"]',
        '[itemprop="articleBody"]',
        ".article",
        ".post",
        ".entry-content",
        ".content",
        ".page-content",
    ):
        node = soup.select_one(selector)
        if not node:
            continue
        txt = normalize_whitespace(node.get_text(" ", strip=True))
        if len(txt) >= 120:
            candidates.append((len(txt), node))
    if candidates:
        return sorted(candidates, key=lambda x: x[0], reverse=True)[0][1]

    body = soup.body or soup
    best_node = body
    best_score = 0.0

    blocks = body.find_all(["article", "section", "div"], recursive=True)
    for block in blocks[:700]:
        txt = normalize_whitespace(block.get_text(" ", strip=True))
        if len(txt) < 180:
            continue
        link_text = normalize_whitespace(" ".join(a.get_text(" ", strip=True) for a in block.find_all("a")))
        link_density = len(link_text) / (len(txt) + 1)
        score = len(txt) * (1 - min(link_density, 0.95))
        if score > best_score:
            best_score = score
            best_node = block
    return best_node


def extract_clean_text(page_soup):
    soup = BeautifulSoup(str(page_soup), "html.parser")

    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()

    for tag in soup.find_all([
        "script", "style", "noscript", "template", "svg", "canvas", "iframe",
        "form", "object", "embed", "link", "meta", "picture", "source"
    ]):
        tag.decompose()

    for tag in soup.find_all(["pre", "code", "kbd", "samp"]):
        tag.decompose()

    for tag in soup.find_all(["header", "footer", "nav", "aside"]):
        tag.decompose()

    for tag in soup.find_all(_is_boilerplate_by_attr):
        tag.decompose()

    main_node = _pick_main_content_node(soup)
    text = normalize_whitespace(main_node.get_text(" ", strip=True))
    text = re.sub(r"\s*[|•·]+\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()

    if _looks_like_noise(text):
        alt = soup.select_one("article") or soup.select_one("main") or soup.body
        alt_text = normalize_whitespace(alt.get_text(" ", strip=True)) if alt else ""
        if alt_text and not _looks_like_noise(alt_text):
            text = alt_text

    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS].rstrip()

    return text


def _safe_csv_value(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    s = str(value)
    s = s.replace("\x00", "")
    return s


def _requests_fetch(url: str) -> str:
    try:
        r = requests.get(url, timeout=12, headers=REQUEST_HEADERS, allow_redirects=True)
        if r.status_code >= 400:
            return ""
        return r.text or ""
    except Exception:
        return ""


def _driver_fetch(driver, url: str) -> str:
    try:
        driver.set_page_load_timeout(18)
    except Exception:
        pass

    for attempt in range(1, 4):
        try:
            driver.get(url)
            time.sleep(0.8)

            try:
                WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except Exception:
                pass

            src = driver.page_source or ""
            if len(src) < 800:
                time.sleep(0.6)
            else:
                return src
        except Exception:
            time.sleep(0.8 * attempt)

    return ""


def _head_status(url: str):
    try:
        r = requests.head(url, timeout=10, allow_redirects=True, headers=REQUEST_HEADERS)
        return r.status_code, r.headers
    except Exception:
        return None, {}


def parse_word_month_date(text):
    if not text:
        return None
    pattern = re.compile(
        r"(?<!\d)(\d{1,2})\s+([A-Za-zА-Яа-яЁё\.]+)\s*,?\s*(20\d{2})(?!\d)",
        re.IGNORECASE
    )
    m = pattern.search(text)
    if not m:
        return None
    day = int(m.group(1))
    month_raw = m.group(2).strip(".").lower()
    year = int(m.group(3))
    month = MONTHS.get(month_raw)
    if not month:
        return None
    try:
        return date(year, month, day).isoformat()
    except Exception:
        return None


def parse_date_any(value):
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()

    s = normalize_whitespace(value)
    if not s:
        return None

    if re.fullmatch(r"\d{10,13}", s):
        try:
            ts = int(s[:10])
            return datetime.utcfromtimestamp(ts).date().isoformat()
        except Exception:
            pass

    try:
        return parsedate_to_datetime(s).date().isoformat()
    except Exception:
        pass

    s_iso = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s_iso).date().isoformat()
    except Exception:
        pass

    for fmt in (
        "%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d",
        "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y",
        "%d %m %Y"
    ):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except Exception:
            pass

    dt = parse_word_month_date(s)
    if dt:
        return dt

    token_patterns = [
        r"(20\d{2}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01]))",
        r"((?:0?[1-9]|[12]\d|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.]20\d{2})",
        r"((?:0?[1-9]|[12]\d|3[01])\s+[A-Za-zА-Яа-яЁё\.]+\s+20\d{2})",
    ]
    for pat in token_patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            dt = parse_date_any(m.group(1))
            if dt:
                return dt

    return None


def extract_date_tokens_from_text(text, limit=5):
    text = normalize_whitespace(text)
    if not text:
        return []
    patterns = [
        r"20\d{2}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])",
        r"(?:0?[1-9]|[12]\d|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.]20\d{2}",
        r"(?:0?[1-9]|[12]\d|3[01])\s+[A-Za-zА-Яа-яЁё\.]+\s+20\d{2}",
    ]
    found = []
    seen = set()
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            token = m.group(0)
            if token in seen:
                continue
            seen.add(token)
            found.append(token)
            if len(found) >= limit:
                return found
    return found


def choose_best_candidate(candidates):
    if not candidates:
        return None, "none"
    best = sorted(
        candidates,
        key=lambda x: (CONF_SCORE.get(x[1], 0), SOURCE_PRIORITY.get(x[2], 0)),
        reverse=True
    )[0]
    return best[0], best[1]


def max_confidence(c1, c2):
    return c1 if CONF_SCORE.get(c1, 0) >= CONF_SCORE.get(c2, 0) else c2


def extract_date_from_url(url):
    path = urlparse(url).path

    m = re.search(r"/((?:19|20)\d{2})/(0?[1-9]|1[0-2])/([0-3]?\d)(?:/|$)", path)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return date(y, mo, d).isoformat()
        except Exception:
            pass

    for pat in (
        r"((?:19|20)\d{2}[-_.](?:0?[1-9]|1[0-2])[-_.](?:[0-3]?\d))",
        r"((?:[0-3]?\d)[-_.](?:0?[1-9]|1[0-2])[-_.](?:19|20)\d{2})",
    ):
        m = re.search(pat, path)
        if m:
            dt = parse_date_any(m.group(1))
            if dt:
                return dt

    return None


def extract_http_last_modified(url):
    try:
        r = requests.head(url, timeout=8, allow_redirects=True, headers=REQUEST_HEADERS)
        lm = r.headers.get("Last-Modified")
        if lm:
            dt = parse_date_any(lm)
            if dt:
                return dt, "medium"
    except Exception:
        pass
    return None, None


def walk_json(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk_json(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from walk_json(item)


def parse_jsonld(raw):
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        return [json.loads(raw)]
    except Exception:
        pass
    objs = []
    for chunk in re.split(r"\n{2,}", raw):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            objs.append(json.loads(chunk))
        except Exception:
            continue
    return objs


def extract_dates(soup, url):
    created_candidates = []
    modified_candidates = []

    def add_created(value, confidence, source):
        dt = parse_date_any(value)
        if dt:
            created_candidates.append((dt, confidence, source))

    def add_modified(value, confidence, source):
        dt = parse_date_any(value)
        if dt:
            modified_candidates.append((dt, confidence, source))

    for meta in soup.find_all("meta"):
        prop = (meta.get("property") or meta.get("name") or meta.get("itemprop") or "").strip().lower()
        content = meta.get("content") or meta.get("value")
        if not prop or not content:
            continue

        if prop in PUBLISHED_KEYS or any(k in prop for k in ("publish", "published", "pubdate", "datecreated")):
            conf = "very_high" if prop in PUBLISHED_KEYS else "high"
            add_created(content, conf, "meta")

        if prop in MODIFIED_KEYS or any(k in prop for k in ("modified", "updated", "lastmod", "last-modified")):
            conf = "very_high" if prop in MODIFIED_KEYS else "high"
            add_modified(content, conf, "meta")

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text()
        for parsed in parse_jsonld(raw):
            for node in walk_json(parsed):
                if not isinstance(node, dict):
                    continue
                node_type = normalize_whitespace(node.get("@type", "")).lower()
                type_boost = "very_high" if any(t in node_type for t in ("article", "newsarticle", "blogposting")) else "high"
                for key, value in node.items():
                    k = str(key).strip().lower()
                    if value in (None, ""):
                        continue
                    if k in ("datepublished", "datecreated", "publishdate", "pubdate", "uploaddate", "date"):
                        add_created(value, type_boost, "jsonld")
                    elif k in ("datemodified", "dateupdated", "modified", "lastmodified", "lastmod"):
                        add_modified(value, type_boost, "jsonld")

    for t in soup.find_all("time"):
        dt_attr = t.get("datetime") or t.get("content")
        classes = " ".join(t.get("class", []))
        ident = t.get("id", "")
        hint_blob = f"{classes} {ident} {t.get_text(' ', strip=True)}"
        if dt_attr:
            if UPDATED_HINT_RE.search(hint_blob):
                add_modified(dt_attr, "high", "time")
            else:
                add_created(dt_attr, "high", "time")
        txt = t.get_text(" ", strip=True)
        if txt:
            if UPDATED_HINT_RE.search(hint_blob):
                add_modified(txt, "medium", "time")
            else:
                add_created(txt, "medium", "time")

    checked = 0
    for el in soup.find_all(True):
        if checked > 250:
            break
        attrs_blob = " ".join([
            el.get("id", ""),
            " ".join(el.get("class", [])),
            el.get("itemprop", ""),
            el.get("aria-label", ""),
            el.get("data-testid", ""),
        ])
        if not DATE_HINT_RE.search(attrs_blob):
            continue
        checked += 1

        for attr in (
            "datetime", "content", "data-date", "data-datetime", "data-time",
            "data-published", "data-publish-date", "data-created",
            "data-modified", "data-lastmod"
        ):
            if el.has_attr(attr):
                if UPDATED_HINT_RE.search(attrs_blob):
                    add_modified(el.get(attr), "medium", "date_hint")
                else:
                    add_created(el.get(attr), "medium", "date_hint")

        txt = normalize_whitespace(el.get_text(" ", strip=True))
        if txt:
            if UPDATED_HINT_RE.search(attrs_blob + " " + txt[:80]):
                add_modified(txt, "medium", "date_hint")
            else:
                add_created(txt, "medium", "date_hint")

    url_dt = extract_date_from_url(url)
    if url_dt:
        add_created(url_dt, "medium", "url")

    http_dt, http_conf = extract_http_last_modified(url)
    if http_dt:
        add_modified(http_dt, http_conf or "low", "http")

    visible_text = normalize_whitespace(soup.get_text(" ", strip=True))[:2500]
    for token in extract_date_tokens_from_text(visible_text, limit=2):
        add_created(token, "low", "text")

    created, created_conf = choose_best_candidate(created_candidates)
    modified, modified_conf = choose_best_candidate(modified_candidates)
    date_confidence = max_confidence(created_conf, modified_conf)

    return created, modified, date_confidence


def _download_pdf(pdf_url: str, out_dir: str, seen_pdf_urls: set):
    if not pdf_url or pdf_url in seen_pdf_urls:
        return False, ""
    seen_pdf_urls.add(pdf_url)

    try:
        head = requests.head(pdf_url, timeout=10, allow_redirects=True, headers=REQUEST_HEADERS)
        ctype = (head.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        clen = head.headers.get("Content-Length")
        if clen:
            try:
                if int(clen) > PDF_MAX_BYTES:
                    return False, ""
            except Exception:
                pass

        if ctype and (ctype not in PDF_ALLOWED_MIME) and (not _looks_like_pdf_url(pdf_url)):
            return False, ""

        r = requests.get(pdf_url, timeout=25, allow_redirects=True, headers=REQUEST_HEADERS, stream=True)
        if r.status_code >= 400:
            return False, ""

        r_ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if r_ctype and (r_ctype not in PDF_ALLOWED_MIME) and (not _looks_like_pdf_url(pdf_url)):
            return False, ""

        os.makedirs(out_dir, exist_ok=True)

        base_name = _pdf_filename_from_url(pdf_url)
        h = hashlib.sha1(pdf_url.encode("utf-8", errors="ignore")).hexdigest()[:10]
        file_name = f"{os.path.splitext(base_name)[0]}_{h}.pdf"
        file_path = os.path.join(out_dir, file_name)

        total = 0
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > PDF_MAX_BYTES:
                    try:
                        f.close()
                    except Exception:
                        pass
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
                    return False, ""
                f.write(chunk)

        if total > 1000:
            return True, file_path
        return False, ""
    except Exception:
        return False, ""


def get_pdf_links_from_soup(soup, page_url):
    pdfs = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = normalize_url(urljoin(page_url, a["href"]))
        if not href or href in seen:
            continue
        if not _looks_like_pdf_url(href):
            continue
        seen.add(href)
        pdfs.append(href)
    return pdfs


def _iso_to_date(iso_str):
    if not iso_str:
        return None
    try:
        return datetime.strptime(iso_str, "%Y-%m-%d").date()
    except Exception:
        return None


def is_news_within_window(created_iso, modified_iso, cutoff_date):
    cd = _iso_to_date(created_iso)
    md = _iso_to_date(modified_iso)
    effective = cd or md
    if not effective:
        return False
    return effective >= cutoff_date


def _is_news_like(url: str, title: str, esg_topic: str) -> bool:
    blob = " ".join([(url or "").lower(), (title or "").lower(), (esg_topic or "").lower()])
    if "/news" in blob or "/press" in blob or "/media" in blob:
        return True
    if "новост" in blob or "пресс" in blob:
        return True
    return False


def scrape_company_links_task(args):
    (
        links_csv_path,
        company_id,
        inn,
        results_dir,
        lock,
        pdf_mode,
        news_mode,
        news_cutoff_date,
        stats,
    ) = args

    parsing_date = datetime.utcnow().date().isoformat()

    try:
        df = pd.read_csv(links_csv_path)
    except Exception:
        df = pd.read_csv(links_csv_path, sep=";")

    canonical_required = {
        "url": "URL",
        "page_title": "page_title",
        "link_text": "link_text",
        "esg_topic": "ESG_topic",
    }

    current_cols_lower = {str(c).strip().lower(): c for c in df.columns}
    missing = [src for src in canonical_required if src not in current_cols_lower]
    if missing:
        return

    rename_map = {}
    for lower_name, canonical_name in canonical_required.items():
        current_name = current_cols_lower[lower_name]
        if current_name != canonical_name:
            rename_map[current_name] = canonical_name

    if rename_map:
        df = df.rename(columns=rename_map)

    with lock:
        stats["total_input_rows"] += int(len(df))

    out_file = os.path.join(results_dir, f"{company_id}.csv")
    os.makedirs(results_dir, exist_ok=True)

    seen_pdf_urls = set()
    pdf_downloaded_count = 0
    pdf_out_dir = os.path.join(results_dir, "pdfs", company_id)

    driver = None
    try:
        driver = get_driver()
    except Exception:
        driver = None

    rows_out = []

    try:
        for _, r in df.iterrows():
            raw_url = r.get("URL")
            url = normalize_url(raw_url)
            if not url:
                with lock:
                    stats["rows_skipped_badurl"] += 1
                continue

            agent_page_title = normalize_whitespace(r.get("page_title"))
            agent_link_text = normalize_whitespace(r.get("link_text"))
            esg_topic = normalize_whitespace(r.get("ESG_topic"))

            try:
                http_status, _hdrs = _head_status(url)

                html_src = ""
                if driver is not None:
                    html_src = _driver_fetch(driver, url)

                if not html_src or len(html_src) < 800:
                    html_src = _requests_fetch(url)

                if not html_src or len(html_src) < 800:
                    with lock:
                        stats["pages_failed_fetch"] += 1
                        stats["rows_failed"] += 1
                    continue

                final_url = (driver.current_url if driver is not None else url) or url
                soup = BeautifulSoup(html_src, "html.parser")

                page_title = ""
                if driver is not None and getattr(driver, "title", None):
                    page_title = driver.title or ""

                if not page_title:
                    page_title = soup.title.get_text(" ", strip=True) if soup.title else ""

                pdf_saved = []
                if pdf_mode and pdf_downloaded_count < PDF_MAX_PER_COMPANY:
                    pdf_links = get_pdf_links_from_soup(soup, final_url)
                    for pdf_url in pdf_links:
                        if pdf_downloaded_count >= PDF_MAX_PER_COMPANY:
                            break
                        ok, saved_path = _download_pdf(pdf_url, pdf_out_dir, seen_pdf_urls)
                        with lock:
                            if ok:
                                stats["pdf_downloaded"] += 1
                            else:
                                stats["pdf_skipped"] += 1
                        if ok:
                            pdf_downloaded_count += 1
                            if saved_path:
                                pdf_saved.append(saved_path)

                clean_text = extract_clean_text(soup)

                if not clean_text or len(clean_text) < MIN_TEXT_CHARS:
                    with lock:
                        stats["rows_skipped_short_text"] += 1
                    continue

                if _looks_like_noise(clean_text):
                    with lock:
                        stats["rows_skipped_noise"] += 1
                    continue

                created, modified, date_confidence = extract_dates(soup, final_url)

                if news_mode and _is_news_like(final_url, page_title, esg_topic):
                    if not is_news_within_window(created, modified, news_cutoff_date):
                        continue

                row = [
                    _safe_csv_value(company_id),
                    _safe_csv_value(inn),
                    _safe_csv_value(url),
                    _safe_csv_value(final_url),
                    _safe_csv_value(agent_page_title),
                    _safe_csv_value(agent_link_text),
                    _safe_csv_value(esg_topic),
                    _safe_csv_value(page_title),
                    _safe_csv_value(http_status),
                    _safe_csv_value(created),
                    _safe_csv_value(modified),
                    _safe_csv_value(date_confidence),
                    _safe_csv_value(parsing_date),
                    _safe_csv_value(clean_text),
                    _safe_csv_value(";".join(pdf_saved) if pdf_saved else ""),
                ]
                rows_out.append(row)

            except Exception:
                with lock:
                    stats["rows_failed"] += 1
                continue

        if not rows_out:
            with lock:
                stats["company_files_done"] += 1
            return

        header = [
            "company_id",
            "INN",
            "URL",
            "final_url",
            "page_title_agent",
            "link_text_agent",
            "ESG_topic",
            "page_title_parsed",
            "http_status",
            "Created",
            "LastModified",
            "DateConfidence",
            "ParsingDate",
            "web_page_text",
            "pdf_saved_paths",
        ]

        with lock:
            write_header = not os.path.exists(out_file)
            with open(out_file, "a", encoding="utf-8", newline="") as f:
                w = csv.writer(
                    f,
                    quoting=csv.QUOTE_MINIMAL,
                    quotechar='"',
                    escapechar="\\",
                    doublequote=True,
                    lineterminator="\n",
                )
                if write_header:
                    w.writerow(header)
                w.writerows(rows_out)

            stats["total_rows_saved"] += int(len(rows_out))
            stats["company_files_done"] += 1

    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass
