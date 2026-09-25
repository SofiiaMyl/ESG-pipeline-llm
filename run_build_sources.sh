#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

source venv/bin/activate
set -a
source ~/.env
[ -f "$PROJECT_DIR/.env" ] && source "$PROJECT_DIR/.env"
set +a

mkdir -p "$PARSED_RESULTS_DIR" "$PARSED_SOURCES_DIR" "$LOG_DIR"

upload_prepared_outputs() {
python3 - <<'PY'
import os
import boto3

bucket = os.environ["S3_BUCKET"]
endpoint = os.environ.get("S3_ENDPOINT", "https://storage.yandexcloud.net")

prepared_dir = os.environ["PARSED_SOURCES_DIR"]
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

upload_tree(prepared_dir, "prepared/parsed_sources")

build_log = os.environ.get("BUILD_SOURCES_LOG", "")
if build_log and os.path.isfile(build_log):
    key = "logs/pipeline/" + os.path.basename(build_log)
    s3.upload_file(build_log, bucket, key)
    print("uploaded:", build_log, "->", key)
PY
}

prepare_parsed_results_subset() {
    if [[ -z "${PARSED_RESULTS_ONLY:-}" && -z "${PARSED_RESULTS_LIST_FILE:-}" ]]; then
        return
    fi

    TMP_DIR="$PROJECT_DIR/data/_parsed_results_subset"
    rm -rf "$TMP_DIR"
    mkdir -p "$TMP_DIR"

    if [[ -n "${PARSED_RESULTS_ONLY:-}" ]]; then
        IFS=',' read -ra ITEMS <<< "$PARSED_RESULTS_ONLY"
        for item in "${ITEMS[@]}"; do
            item="$(echo "$item" | xargs)"
            [[ -z "$item" ]] && continue
            cp "$PARSED_RESULTS_DIR/$item" "$TMP_DIR/"
        done
    fi

    if [[ -n "${PARSED_RESULTS_LIST_FILE:-}" && -f "${PARSED_RESULTS_LIST_FILE}" ]]; then
        while IFS= read -r item; do
            item="$(echo "$item" | xargs)"
            [[ -z "$item" ]] && continue
            cp "$PARSED_RESULTS_DIR/$item" "$TMP_DIR/"
        done < "$PARSED_RESULTS_LIST_FILE"
    fi

    export PARSED_RESULTS_DIR="$TMP_DIR"
    echo "subset PARSED_RESULTS_DIR=$PARSED_RESULTS_DIR"
}

prepare_parsed_results_subset

python3 adapters/build_sources.py

upload_prepared_outputs
echo "run_build_sources.sh done"
