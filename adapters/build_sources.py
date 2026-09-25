import os
import re
import csv
import hashlib
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

BASE_DATA_DIR = os.environ.get(
    "PIPELINE_DATA_DIR",
    "/home/ubuntu/esg-pipeline/data"
)

PARSED_RESULTS_DIR = os.environ.get(
    "PARSED_RESULTS_DIR",
    os.path.join(BASE_DATA_DIR, "parsed_results")
)

PARSED_SOURCES_DIR = os.environ.get(
    "PARSED_SOURCES_DIR",
    os.path.join(BASE_DATA_DIR, "parsed_sources")
)

BUILD_SOURCES_LOG = os.environ.get(
    "BUILD_SOURCES_LOG",
    os.path.join(BASE_DATA_DIR, "logs", "build_sources.log")
)

MIN_TEXT_CHARS = int(os.environ.get("BUILD_SOURCES_MIN_TEXT_CHARS", "120"))
MAX_TEXT_CHARS = int(os.environ.get("BUILD_SOURCES_MAX_TEXT_CHARS", "40000"))

EXTERNAL_SOURCES_PREFIX = "esg_urls_"

def norm_ws(text):
    if text is None:
        return ""
    text = str(text).replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def safe_str(val):
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except Exception:
        pass
    return norm_ws(val)

def read_csv_flexible(path):
    for enc in ("utf-8-sig", "utf-8", "cp1251"):
        for sep in (",", ";"):
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc, dtype=str).fillna("")
                if df.shape[1] >= 3:
                    return df
            except Exception:
                continue
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", dtype=str).fillna("")

def domain_from_url(url):
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""

def safe_name(s):
    s = safe_str(s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "item"

def cut_text(text, limit=MAX_TEXT_CHARS):
    text = norm_ws(text)
    if len(text) > limit:
        return text[:limit].rstrip()
    return text

def text_from_pdf(pdf_path):
    if PdfReader is None:
        return ""
    if not pdf_path or not os.path.isfile(pdf_path):
        return ""

    parts = []
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            txt = norm_ws(txt)
            if txt:
                parts.append(txt)
        return cut_text(" ".join(parts))
    except Exception:
        return ""

def split_pdf_paths(raw):
    raw = safe_str(raw)
    if not raw:
        return []
    parts = [x.strip() for x in raw.split(";")]
    parts = [x for x in parts if x]
    return parts

def write_text_file(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def log_line(msg):
    os.makedirs(os.path.dirname(BUILD_SOURCES_LOG), exist_ok=True)
    with open(BUILD_SOURCES_LOG, "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")

def _infer_esg_topic_from_filename(filename: str) -> str:
    """
    Определить ESG-тему из имени файла.
    Примеры: "eco_report.txt" → "environment", "social_policy.txt" → "social"
    """
    lower = filename.lower()

    if any(k in lower for k in ["eco", "env", "green", "carbon", "emission", "climate"]):
        return "environment"
    if any(k in lower for k in ["social", "hr", "employee", "community", "diversity"]):
        return "social"
    if any(k in lower for k in ["gov", "board", "ethic", "compliance", "audit"]):
        return "governance"

    return ""  

def _find_external_pdfs(company_id: str) -> list[str]:
    """
    Найти PDF-файлы для компании, загруженные локальным скриптом вне CSV.
    Основной путь: ${PARSED_SOURCES_DIR}/esg_urls_<company_id>/*.pdf
    Дополнительно (backward-compat) проверяем старую схему на случай,
    если где-то остались файлы со старого запуска.
    """
    search_paths = []

    ext_dir = _external_sources_dir(company_id)
    if ext_dir.exists():
        search_paths.append(ext_dir)

    old_dir = Path(PARSED_RESULTS_DIR) / "pdfs" / company_id
    if old_dir.exists():
        search_paths.append(old_dir)

    found_pdfs = []
    for directory in search_paths:
        for pdf_path in directory.glob("*.pdf"):
            found_pdfs.append(str(pdf_path.resolve()))
    return found_pdfs

def _company_id_from_csv_stem(csv_path: Path) -> str:
    """
    Единый company_id = имя CSV-файла без префикса esg_urls_ и расширения.
    Локальный скрипт загружает CSV как esg_urls_<canonical_id>.csv,
    PDF/TXT — в папки esg_urls_<canonical_id>/ — поэтому имя файла
    надёжнее, чем внутренняя колонка company_id в самом CSV (она может
    быть меткой отчёта/партии, а не устойчивым id компании).
    """
    stem = csv_path.stem
    if stem.startswith(EXTERNAL_SOURCES_PREFIX):
        return stem[len(EXTERNAL_SOURCES_PREFIX):]
    return safe_name(stem)

def _process_external_txt_sources(index_rows, stats, global_seq, company_id):
    """
    Обработать TXT-файлы, загруженные внешне (не из CSV парсинга), для
    КОНКРЕТНОЙ компании. Ищет .txt в ${PARSED_SOURCES_DIR}/esg_urls_<company_id>/ —
    той же папке, куда build_sources кладёт и внешние PDF.
    """
    existing_files = {r.get("file_path", "") for r in index_rows}

    ext_dir = _external_sources_dir(company_id)
    if not ext_dir.is_dir():
        return global_seq

    company_dir = Path(PARSED_SOURCES_DIR) / company_id
    company_dir.mkdir(parents=True, exist_ok=True)

    for txt_path in sorted(ext_dir.glob("*.txt")):
        try:
            text = txt_path.read_text(encoding="utf-8")
        except Exception as e:
            log_line(f"ERROR reading {txt_path}: {e}")
            continue

        text_len = len(text.strip())
        if text_len < MIN_TEXT_CHARS:
            continue

        dest_name = f"source_{global_seq:06d}_txt_external_{safe_name(txt_path.stem)}.txt"
        rel_path = f"{company_id}/{dest_name}"

        if rel_path in existing_files:
            continue

        write_text_file(str(company_dir / dest_name), text)

        inferred_topic = _infer_esg_topic_from_filename(txt_path.name)
        source_id = f"{company_id}_txt_{global_seq:06d}"

        index_rows.append({
            "company_id": company_id,
            "inn": "",
            "domain": "",
            "source_id": source_id,
            "source_type": "txt_external",
            "url": "",
            "final_url": str(txt_path),
            "page_title": txt_path.stem,
            "link_text": "",
            "esg_topic": inferred_topic,
            "http_status": "",
            "created": "",
            "last_modified": "",
            "date_confidence": "",
            "parsing_date": "",
            "file_path": rel_path,
            "text_chars": text_len,
        })

        stats["txt_external_sources_saved"] += 1
        global_seq += 1

    return global_seq

def _external_sources_dir(company_id: str) -> Path:
    """
    Общая папка, куда локальный скрипт скачивает и PDF, и TXT для компании:
    ${PARSED_SOURCES_DIR}/esg_urls_<company_id>/
    """
    return Path(PARSED_SOURCES_DIR) / f"{EXTERNAL_SOURCES_PREFIX}{company_id}"

def _find_external_pdfs(company_id: str) -> list[str]:
    """
    Найти PDF-файлы для компании вне CSV.
    Ищет в:
        1. $PARSED_RESULTS_DIR/pdfs/<company_id>/
        2. $PARSED_RESULTS_DIR/../pdfs/<company_id>/
    """
    search_paths = []
    
    ext_dir = _external_sources_dir(company_id)
    if ext_dir.exists():
        search_paths.append(ext_dir)

    old_dir = Path(PARSED_RESULTS_DIR) / "pdfs" / company_id
    if old_dir.exists():
        search_paths.append(old_dir)
    
    found_pdfs = []
    for directory in search_paths:
        for pdf_path in directory.glob("*.pdf"):
            found_pdfs.append(str(pdf_path.resolve()))
    
    return found_pdfs

def build_sources():
    os.makedirs(PARSED_SOURCES_DIR, exist_ok=True)

    files = sorted(Path(PARSED_RESULTS_DIR).glob("*.csv"))
    if not files:
        print(f"No parsed result CSV files found in {PARSED_RESULTS_DIR}")
        return

    index_rows = []

    stats = {
        "company_files": 0,
        "web_sources_saved": 0,
        "pdf_sources_saved": 0,
        "rows_skipped_short": 0,
        "pdf_skipped": 0,
        "txt_external_sources_saved": 0, #счетчик внешних txt
        "pdf_external_sources_saved": 0, # внешние pdf
    }
    global_seq = 1 # глобальная нумерация всех источников

    for csv_path in files:
        df = read_csv_flexible(str(csv_path))
        if df.empty:
            continue

        stats["company_files"] += 1

        company_id = _company_id_from_csv_stem(csv_path)
        company_dir = Path(PARSED_SOURCES_DIR) / company_id
        company_dir.mkdir(parents=True, exist_ok=True)

        # seq = 1
        all_pdf_paths = set()
        last_row_meta = {}

        for _, row in df.iterrows():
            inn = safe_str(row.get("INN"))
            url = safe_str(row.get("URL"))
            final_url = safe_str(row.get("final_url")) or url
            domain = domain_from_url(final_url or url)
            page_title = safe_str(row.get("page_title_parsed")) or safe_str(row.get("page_title_agent"))
            link_text = safe_str(row.get("link_text_agent"))
            esg_topic = safe_str(row.get("ESG_topic"))
            http_status = safe_str(row.get("http_status"))
            created = safe_str(row.get("Created"))
            last_modified = safe_str(row.get("LastModified"))
            date_conf = safe_str(row.get("DateConfidence"))
            parsing_date = safe_str(row.get("ParsingDate"))
            web_text = cut_text(row.get("web_page_text"))

            last_row_meta = {
                "inn": inn, "domain": domain, "page_title": page_title,
                "link_text": link_text, "esg_topic": esg_topic,
                "created": created, "last_modified": last_modified,
                "date_conf": date_conf, "parsing_date": parsing_date,
            }

            if len(web_text) >= MIN_TEXT_CHARS:
                source_id = f"{company_id}_web_{global_seq:06d}"
                file_name = f"source_{global_seq:06d}_web.txt"
                rel_path = f"{company_id}/{file_name}"
                abs_path = str(company_dir / file_name)

                write_text_file(abs_path, web_text)

                index_rows.append({
                    "company_id": company_id,
                    "inn": inn,
                    "domain": domain,
                    "source_id": source_id,
                    "source_type": "web",
                    "url": url,
                    "final_url": final_url,
                    "page_title": page_title,
                    "link_text": link_text,
                    "esg_topic": esg_topic,
                    "http_status": http_status,
                    "created": created,
                    "last_modified": last_modified,
                    "date_confidence": date_conf,
                    "parsing_date": parsing_date,
                    "file_path": rel_path,
                    "text_chars": len(web_text),
                })

                stats["web_sources_saved"] += 1
                global_seq+=1
            else:
                stats["rows_skipped_short"] += 1

            all_pdf_paths.update(split_pdf_paths(row.get("pdf_saved_paths")))

        external_pdfs = set(_find_external_pdfs(company_id))
        if external_pdfs:
                print(f"  [company: {company_id}] Найдено {len(external_pdfs)} внешних PDF")
        all_pdf_paths.update(external_pdfs)

        for pdf_path in all_pdf_paths:
                pdf_text = text_from_pdf(pdf_path)
                if len(pdf_text) < MIN_TEXT_CHARS:
                    stats["pdf_skipped"] += 1
                    continue

                pdf_hash = hashlib.sha1(pdf_path.encode("utf-8", errors="ignore")).hexdigest()[:10]
                source_id = f"{company_id}_pdf_{global_seq:06d}"
                file_name = f"source_{global_seq:06d}_pdf_{pdf_hash}.txt"
                rel_path = f"{company_id}/{file_name}"
                abs_path = str(company_dir / file_name)

                write_text_file(abs_path, pdf_text)

                source_type = "pdf_external" if pdf_path in external_pdfs else "pdf"

                index_rows.append({
                "company_id": company_id,
                "inn": last_row_meta.get("inn", ""),
                "domain": last_row_meta.get("domain", ""),
                "source_id": source_id,
                "source_type": source_type,  # исправлено: раньше всегда "pdf"
                "url": "", "final_url": pdf_path,
                "page_title": last_row_meta.get("page_title", ""),
                "link_text": last_row_meta.get("link_text", ""),
                "esg_topic": last_row_meta.get("esg_topic", ""),
                "http_status": "",
                "created": last_row_meta.get("created", ""),
                "last_modified": last_row_meta.get("last_modified", ""),
                "date_confidence": last_row_meta.get("date_conf", ""),
                "parsing_date": last_row_meta.get("parsing_date", ""),
                "file_path": rel_path, "text_chars": len(pdf_text),
                })

                if source_type == "pdf_external":
                    stats["pdf_external_sources_saved"] += 1
                else:
                    stats["pdf_sources_saved"] += 1

                global_seq+=1

        # обработка txt
        global_seq = _process_external_txt_sources(index_rows, stats, global_seq, company_id)

        index_path = Path(PARSED_SOURCES_DIR) / "_sources_index.csv"
        pd.DataFrame(index_rows).to_csv(index_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

        log_line("=" * 70)
        for k, v in stats.items():
            log_line(f"{k}: {v}")
        log_line(f"index_path: {index_path}")
        log_line("=" * 70)

        print("Done.")
        print(f"Sources index written to: {index_path}")
        print(f"Web sources: {stats['web_sources_saved']}")
        print(f"PDF sources: {stats['pdf_sources_saved']}")
        print(f"PDF sources (external): {stats['pdf_external_sources_saved']}")
        print(f"TXT external sources: {stats['txt_external_sources_saved']}")

if __name__ == "__main__":
    build_sources()
