#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

source venv/bin/activate
set -a
source ~/.env
[ -f "$PROJECT_DIR/.env" ] && source "$PROJECT_DIR/.env"
set +a

UPLOAD_PREPARED=0
UPLOAD_OUTPUT=0
UPLOAD_LOGS=0
UPLOAD_INTERMEDIATE=0

if [[ $# -eq 0 ]]; then
    UPLOAD_PREPARED=1
    UPLOAD_OUTPUT=1
    UPLOAD_LOGS=1
    UPLOAD_INTERMEDIATE=1
else
    for arg in "$@"; do
        case "$arg" in
            --prepared)
                UPLOAD_PREPARED=1
                ;;
            --output)
                UPLOAD_OUTPUT=1
                ;;
            --logs)
                UPLOAD_LOGS=1
                ;;
            --intermediate)
                UPLOAD_INTERMEDIATE=1
                ;;
            *)
                echo "Unknown option: $arg"
                echo "Usage: ./run_upload_results.sh [--prepared] [--output] [--logs] [--intermediate]"
                exit 1
                ;;
        esac
    done
fi

export UPLOAD_PREPARED
export UPLOAD_OUTPUT
export UPLOAD_LOGS
export UPLOAD_INTERMEDIATE

python3 - <<'PY'
import os
import boto3

bucket = os.environ["S3_BUCKET"]
endpoint = os.environ.get("S3_ENDPOINT", "https://storage.yandexcloud.net")

prepared_dir = os.environ.get("PARSED_SOURCES_DIR", "")
output_dir = os.environ.get("OUTPUT_DIR", "")
log_dir = os.environ.get("LOG_DIR", "")
intermediate_dir = os.environ.get("RAG_INTERMEDIATE_DIR", "")

upload_prepared = os.environ.get("UPLOAD_PREPARED", "0") == "1"
upload_output = os.environ.get("UPLOAD_OUTPUT", "0") == "1"
upload_logs = os.environ.get("UPLOAD_LOGS", "0") == "1"
upload_intermediate = os.environ.get("UPLOAD_INTERMEDIATE", "0") == "1"

s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ.get("AWS_DEFAULT_REGION", "ru-central1"),
)

def upload_tree(local_root, remote_prefix):
    if not local_root:
        print(f"skip empty path for {remote_prefix}")
        return
    if not os.path.isdir(local_root):
        print(f"skip missing dir: {local_root}")
        return

    uploaded = 0
    for root, _, files in os.walk(local_root):
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, local_root).replace("\\", "/")
            key = f"{remote_prefix}/{rel}"
            s3.upload_file(path, bucket, key)
            print("uploaded:", path, "->", key)
            uploaded += 1

    print(f"done {remote_prefix}: {uploaded} files")

if upload_prepared:
    upload_tree(prepared_dir, "prepared/parsed_sources")

if upload_output:
    upload_tree(output_dir, "output")

if upload_logs:
    upload_tree(log_dir, "logs/pipeline")

if upload_intermediate:
    upload_tree(intermediate_dir, "output/intermediate")
PY
