from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from datetime import datetime
from typing import Optional
import boto3
import os
import json
import io

app = FastAPI()

YC_FOLDER_ID = os.environ["YC_FOLDER_ID"]
YC_AI_API_KEY = os.environ["YC_AI_API_KEY"]
CLIENT_API_KEY = os.environ["CLIENT_API_KEY"]

S3_BUCKET = os.environ["S3_BUCKET"]
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "https://storage.yandexcloud.net")

llm_client = OpenAI(
    api_key=YC_AI_API_KEY,
    base_url="https://llm.api.cloud.yandex.net/v1"
)

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name=os.environ.get("AWS_DEFAULT_REGION", "ru-central1")
)

# ROLE_TO_MODEL = {
#     "scorer": f"gpt://{YC_FOLDER_ID}/qwen3-235b-a22b-fp8/latest",
#     "auditor": f"gpt://{YC_FOLDER_ID}/qwen3-235b-a22b-fp8/latest",
#     "validator": f"gpt://{YC_FOLDER_ID}/qwen3-235b-a22b-fp8/latest",
# }
ROLE_TO_MODEL = {
    "scorer": os.environ.get("MODEL_SCORER", "gpt://b1gq04ro9q7t98ku2bce/qwen3-235b-a22b-fp8/latest"),
    "auditor": os.environ.get("MODEL_AUDITOR", "gpt://b1gq04ro9q7t98ku2bce/qwen3-235b-a22b-fp8/latest"),
    "validator": os.environ.get("MODEL_VALIDATOR", "gpt://b1gq04ro9q7t98ku2bce/qwen3-235b-a22b-fp8/latest"),
}

ROLE_TO_SYSTEM = {
    "scorer": "You are an ESG scorer. Evaluate the input clearly and concisely.",
    "auditor": "You are an ESG auditor. Check evidence, gaps and weak points.",
    "validator": "You are an ESG validator. Check format and unsupported claims."
}


class LlmReq(BaseModel):
    role: str
    prompt: str
    temperature: float = 0.2


class LogReq(BaseModel):
    event: str
    status: str = "ok"
    company: Optional[str] = None
    details: Optional[dict] = None


def check_key(x_api_key):
    if x_api_key != CLIENT_API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/llm/generate")
def generate(req: LlmReq, x_api_key: str = Header(default="")):
    check_key(x_api_key)

    model = ROLE_TO_MODEL.get(req.role)
    system_prompt = ROLE_TO_SYSTEM.get(req.role)

    if not model:
        raise HTTPException(status_code=400, detail="unknown role")

    try:
        resp = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.prompt}
            ],
            temperature=req.temperature
        )

        return {
            "role": req.role,
            "model": model,
            "text": resp.choices[0].message.content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/generate")
def generate(req: LlmReq, x_api_key: str = Header(default="")):
    check_key(x_api_key)

    model = ROLE_TO_MODEL.get(req.role)
    system_prompt = ROLE_TO_SYSTEM.get(req.role)

    if not model:
        raise HTTPException(status_code=400, detail="unknown role")

    try:
        resp = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.prompt}
            ],
            temperature=req.temperature
        )

        return {
            "role": req.role,
            "model": model,
            "text": resp.choices[0].message.content
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/files/upload")
async def upload_file(
    prefix: str,
    file: UploadFile = File(...),
    x_api_key: str = Header(default="")
):
    check_key(x_api_key)

    key = prefix.rstrip("/") + "/" + file.filename
    data = await file.read()

    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=data
        )

        return {
            "status": "uploaded",
            "bucket": S3_BUCKET,
            "key": key
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files/list")
def list_files(prefix: str = "", x_api_key: str = Header(default="")):
    check_key(x_api_key)

    try:
        resp = s3.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=prefix
        )

        files = []
        for obj in resp.get("Contents", []):
            files.append(obj["Key"])

        return {
            "bucket": S3_BUCKET,
            "prefix": prefix,
            "files": files
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files/download")
def download_file(key: str, x_api_key: str = Header(default="")):
    check_key(x_api_key)

    try:
        obj = s3.get_object(
            Bucket=S3_BUCKET,
            Key=key
        )

        data = obj["Body"].read()
        filename = key.split("/")[-1]

        return StreamingResponse(
            io.BytesIO(data),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/logs/write")
def write_log(req: LogReq, x_api_key: str = Header(default="")):
    check_key(x_api_key)

    today = datetime.utcnow().strftime("%Y-%m-%d")
    now = datetime.utcnow().isoformat()

    key = f"logs/gateway/{today}.jsonl"

    record = {
        "timestamp": now,
        "event": req.event,
        "status": req.status,
        "company": req.company,
        "details": req.details or {}
    }

    line = json.dumps(record, ensure_ascii=False) + "\n"

    try:
        old = ""

        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            old = obj["Body"].read().decode("utf-8")
        except Exception:
            old = ""

        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=(old + line).encode("utf-8")
        )

        return {
            "status": "logged",
            "bucket": S3_BUCKET,
            "key": key
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
