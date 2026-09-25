#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

source venv/bin/activate
set -a
source ~/.env
[ -f "$PROJECT_DIR/.env" ] && source "$PROJECT_DIR/.env"
set +a

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$RAG_INTERMEDIATE_DIR"

if [[ $# -ge 1 && -n "${1:-}" ]]; then
    export RAG_COMPANY_FILTER="$1"
fi

upload_rag_outputs() {
python3 - <<PY
import os
import boto3

bucket = os.environ["S3_BUCKET"]
endpoint = os.environ.get("S3_ENDPOINT", "https://storage.yandexcloud.net")

output_dir = os.environ["OUTPUT_DIR"]
log_dir = os.environ["LOG_DIR"]
intermediate_dir = os.environ["RAG_INTERMEDIATE_DIR"]

s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ.get("AWS_DEFAULT_REGION", "ru-central1"),
)

START_TS = $START_TS

def upload_tree(local_root, remote_prefix):
    if not os.path.isdir(local_root):
        return
    uploaded = 0
    skipped = 0
    for root, _, files in os.walk(local_root):
        for name in files:
            path = os.path.join(root, name)
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if mtime < START_TS:
                skipped += 1
                continue
            rel = os.path.relpath(path, local_root).replace("\\\\", "/")
            key = f"{remote_prefix}/{rel}"
            s3.upload_file(path, bucket, key)
            print("uploaded:", path, "->", key)
            uploaded += 1
    print(f"done {remote_prefix}: {uploaded} uploaded, {skipped} skipped (older than this run)")

upload_tree(output_dir, "output")
upload_tree(log_dir, "logs/pipeline")
upload_tree(intermediate_dir, "output/intermediate")
PY
}

RAG_LOG_FILE="${LOG_DIR}/rag_run.log"
START_TS=$(python3 -c "import time; print(time.time())")
python3 rag/rag_by_variable.py 2>&1 | tee "$RAG_LOG_FILE"

export START_TS
upload_rag_outputs
echo "run_rag.sh done"