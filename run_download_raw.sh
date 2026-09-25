#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

source venv/bin/activate
set -a
source ~/.env
[ -f "$PROJECT_DIR/.env" ] && source "$PROJECT_DIR/.env"
set +a

DOWNLOAD_SELECTED=0
DOWNLOAD_PARSED=0
DOWNLOAD_PREPARED=0

if [[ $# -eq 0 ]]; then
    DOWNLOAD_SELECTED=1
else
    for arg in "$@"; do
        case "$arg" in
            --selected)
                DOWNLOAD_SELECTED=1
                ;;
            --parsed)
                DOWNLOAD_PARSED=1
                ;;
            --prepared)
                DOWNLOAD_PREPARED=1
                ;;
            *)
                echo "Unknown option: $arg"
                echo "Usage: ./run_download_raw.sh [--selected] [--parsed] [--prepared]"
                exit 1
                ;;
        esac
    done
fi

export DOWNLOAD_SELECTED
export DOWNLOAD_PARSED
export DOWNLOAD_PREPARED

mkdir -p "$SELECTED_LINKS_DIR" "$PARSED_RESULTS_DIR" "$PARSED_SOURCES_DIR"

python3 - <<'PY'
import os
import boto3

bucket = os.environ["S3_BUCKET"]
endpoint = os.environ.get("S3_ENDPOINT", "https://storage.yandexcloud.net")

selected_dir = os.environ["SELECTED_LINKS_DIR"]
parsed_dir = os.environ["PARSED_RESULTS_DIR"]
prepared_dir = os.environ["PARSED_SOURCES_DIR"]

download_selected = os.environ.get("DOWNLOAD_SELECTED", "0") == "1"
download_parsed = os.environ.get("DOWNLOAD_PARSED", "0") == "1"
download_prepared = os.environ.get("DOWNLOAD_PREPARED", "0") == "1"

s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ.get("AWS_DEFAULT_REGION", "ru-central1"),
)

def download_prefix(prefix, local_root):
    os.makedirs(local_root, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    count = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue

            rel = key[len(prefix):]
            if not rel:
                continue

            out_path = os.path.join(local_root, rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            s3.download_file(bucket, key, out_path)
            print("downloaded:", key, "->", out_path)
            count += 1

    print(f"done {prefix}: {count} files")

if download_selected:
    download_prefix("raw/selected_links/", selected_dir)

if download_parsed:
    download_prefix("raw/parsed_results/", parsed_dir)

if download_prepared:
    download_prefix("prepared/parsed_sources/", prepared_dir)
PY
