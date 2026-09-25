#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

source venv/bin/activate
set -a
source ~/.env
[ -f "$PROJECT_DIR/.env" ] && source "$PROJECT_DIR/.env"
set +a

mkdir -p "$SELECTED_LINKS_DIR" "$PARSED_RESULTS_DIR" "$LOG_DIR"

sync_s3_to_local() {
python3 - <<'PY'
import os
import boto3

bucket = os.environ["S3_BUCKET"]
endpoint = os.environ.get("S3_ENDPOINT", "https://storage.yandexcloud.net")
prefix = "raw/selected_links/"
local_dir = os.environ["SELECTED_LINKS_DIR"]

os.makedirs(local_dir, exist_ok=True)

s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ.get("AWS_DEFAULT_REGION", "ru-central1"),
)

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

        out_path = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        s3.download_file(bucket, key, out_path)
        print("downloaded:", key, "->", out_path)
        count += 1

print("downloaded files:", count)
PY
}

upload_parse_outputs() {
python3 - <<'PY'
import os
import boto3

bucket = os.environ["S3_BUCKET"]
endpoint = os.environ.get("S3_ENDPOINT", "https://storage.yandexcloud.net")

parsed_dir = os.environ["PARSED_RESULTS_DIR"]
log_dir = os.environ["LOG_DIR"]

s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ.get("AWS_DEFAULT_REGION", "ru-central1"),
)

def upload_tree(local_root, remote_prefix):
    if not os.path.isdir(local_root):
        return
    for root, _, files in os.walk(local_root):
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, local_root).replace("\\", "/")
            key = f"{remote_prefix}/{rel}"
            s3.upload_file(path, bucket, key)
            print("uploaded:", path, "->", key)

upload_tree(parsed_dir, "raw/parsed_results")
upload_tree(log_dir, "logs/parser")
PY
}

prepare_selected_links_subset() {
    if [[ -z "${SELECTED_LINKS_ONLY:-}" && -z "${SELECTED_LINKS_LIST_FILE:-}" ]]; then
        return
    fi

    TMP_DIR="$PROJECT_DIR/data/_selected_links_subset"
    rm -rf "$TMP_DIR"
    mkdir -p "$TMP_DIR"

    if [[ -n "${SELECTED_LINKS_ONLY:-}" ]]; then
        IFS=',' read -ra ITEMS <<< "$SELECTED_LINKS_ONLY"
        for item in "${ITEMS[@]}"; do
            item="$(echo "$item" | xargs)"
            [[ -z "$item" ]] && continue
            cp "$SELECTED_LINKS_DIR/$item" "$TMP_DIR/"
        done
    fi

    if [[ -n "${SELECTED_LINKS_LIST_FILE:-}" && -f "${SELECTED_LINKS_LIST_FILE}" ]]; then
        while IFS= read -r item; do
            item="$(echo "$item" | xargs)"
            [[ -z "$item" ]] && continue
            cp "$SELECTED_LINKS_DIR/$item" "$TMP_DIR/"
        done < "$SELECTED_LINKS_LIST_FILE"
    fi

    export SELECTED_LINKS_DIR="$TMP_DIR"
    echo "subset SELECTED_LINKS_DIR=$SELECTED_LINKS_DIR"
}

sync_s3_to_local

if [[ "${DOWNLOAD_ONLY:-0}" == "1" ]]; then
    echo "DOWNLOAD_ONLY=1, stop after sync"
    exit 0
fi

prepare_selected_links_subset

python3 -m parser.main_links

upload_parse_outputs
echo "run_parse.sh done"
