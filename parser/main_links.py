import os
import glob
import multiprocessing
from datetime import datetime, timedelta

from tqdm import tqdm

from .workers_links import scrape_company_links_task

BASE_DATA_DIR = os.environ.get(
    "PIPELINE_DATA_DIR",
    "/home/ubuntu/esg-pipeline/data"
)

INPUT_LINKS_DIR = os.environ.get(
    "SELECTED_LINKS_DIR",
    os.path.join(BASE_DATA_DIR, "selected_links")
)

RESULTS_DIR = os.environ.get(
    "PARSED_RESULTS_DIR",
    os.path.join(BASE_DATA_DIR, "parsed_results")
)

LOG_DIR = os.environ.get(
    "LOG_DIR",
    os.path.join(BASE_DATA_DIR, "logs")
)

LOG_FILE = os.environ.get(
    "PARSER_LOG_FILE",
    os.path.join(LOG_DIR, "parser_run.log")
)

PDF_DOWNLOAD_ENABLED = os.environ.get("PARSER_PDF_ENABLED", "1") == "1"
NEWS_ENABLED = os.environ.get("PARSER_NEWS_ENABLED", "0") == "1"
NEWS_MAX_AGE_DAYS = int(os.environ.get("PARSER_NEWS_MAX_AGE_DAYS", "365"))
MAX_PROCS = int(os.environ.get("PARSER_MAX_PROCS", "8"))


def _safe_company_id_from_filename(path):
    base = os.path.basename(path)
    name = os.path.splitext(base)[0].strip()
    return name or "company"


def write_log(stats, news_cutoff_date):
    os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"Run at: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Input dir: {os.path.abspath(INPUT_LINKS_DIR)}\n")
        f.write(f"Results dir: {os.path.abspath(RESULTS_DIR)}\n")
        f.write(f"PDF enabled: {PDF_DOWNLOAD_ENABLED}\n")
        f.write(f"News enabled: {NEWS_ENABLED}\n")
        f.write(
            f"News cutoff (UTC date): {news_cutoff_date.isoformat()} "
            f"(last {NEWS_MAX_AGE_DAYS} days)\n"
        )
        f.write("-" * 70 + "\n")
        for k in sorted(stats.keys()):
            f.write(f"{k}: {stats[k]}\n")
        f.write("=" * 70 + "\n\n")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(INPUT_LINKS_DIR, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in {INPUT_LINKS_DIR}")
        return

    manager = multiprocessing.Manager()
    lock = manager.Lock()

    stats = manager.dict({
        "total_company_files": 0,
        "company_files_done": 0,
        "total_input_rows": 0,
        "total_rows_saved": 0,
        "rows_failed": 0,
        "rows_skipped_badurl": 0,
        "rows_skipped_short_text": 0,
        "rows_skipped_noise": 0,
        "pdf_downloaded": 0,
        "pdf_skipped": 0,
        "pages_failed_fetch": 0,
    })

    today_utc = datetime.utcnow().date()
    news_cutoff_date = today_utc - timedelta(days=NEWS_MAX_AGE_DAYS)

    tasks = []
    for fp in csv_files:
        company_id = _safe_company_id_from_filename(fp)
        inn = ""

        tasks.append((
            fp,
            company_id,
            inn,
            RESULTS_DIR,
            lock,
            PDF_DOWNLOAD_ENABLED,
            NEWS_ENABLED,
            news_cutoff_date,
            stats,
        ))

    stats["total_company_files"] = len(tasks)

    procs = min(MAX_PROCS, max(1, multiprocessing.cpu_count() - 1))

    with multiprocessing.Pool(processes=procs) as pool:
        list(tqdm(
            pool.imap_unordered(scrape_company_links_task, tasks),
            total=len(tasks)
        ))

    write_log(stats, news_cutoff_date)
    print("Done.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
