"""
main.py  ─  RAG Chatbot API
════════════════════════════
Endpoints:
  POST /order   ← صوت أو نص أو الاثنين معاً
  GET  /health
"""

import os
import uuid
import requests
import traceback
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

from chatbot_chain import chat

load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
ORDERS_API_URL = os.getenv("ORDERS_API_URL", "https://your-app.com/api/orders")

UPLOAD_DIR = "audio_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_AUDIO = {".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac"}
AUDIO_CONTENT_TYPES = {
    ".mp3":  "audio/mpeg",
    ".wav":  "audio/wav",
    ".m4a":  "audio/mp4",
    ".webm": "audio/webm",
    ".ogg":  "audio/ogg",
    ".flac": "audio/flac",
}

groq_client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="RAG Order Chatbot", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Whisper ──────────────────────────────────────────────────────────────────
def transcribe_audio(audio_path: str, language: str = "ar") -> str:
    with open(audio_path, "rb") as f:
        result = groq_client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=f,
            language=language,
            response_format="text",
        )
    return result.strip() if isinstance(result, str) else str(result).strip()


# ─── إرسال نصي للـ API ────────────────────────────────────────────────────────
def send_text_to_api(customer_id: str, text: str, order_data: dict, status: str = "pending") -> dict:
    """
    يبعت JSON:
    { customer_id, text, order, status }
    """
    try:
        response = requests.post(
            ORDERS_API_URL,
            json={
                "customer_id": customer_id,
                "text":        text,
                "order":       order_data,
                "status":      status,
            },
            timeout=10,
        )
        return {"sent": True, "mode": "text", "status_code": response.status_code}
    except Exception as e:
        return {"sent": False, "mode": "text", "error": str(e)}


# ─── إرسال صوتي للـ API ───────────────────────────────────────────────────────
def send_voice_to_api(customer_id: str, audio_path: str, status: str = "pending") -> dict:
    """
    يبعت multipart/form-data:
    { voice: [ملف], customer_id, status }
    """
    try:
        ext          = os.path.splitext(audio_path)[1].lower()
        content_type = AUDIO_CONTENT_TYPES.get(ext, "audio/mpeg")
        with open(audio_path, "rb") as audio_file:
            response = requests.post(
                ORDERS_API_URL,
                files={"voice": (os.path.basename(audio_path), audio_file, content_type)},
                data={"customer_id": customer_id, "status": status},
                timeout=30,
            )
        return {"sent": True, "mode": "voice", "status_code": response.status_code}
    except Exception as e:
        return {"sent": False, "mode": "voice", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
#  GET /health
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "RAG Order Chatbot v2"}


# ─────────────────────────────────────────────────────────────────────────────
#  POST /order  ← endpoint واحد للصوت والنص
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/order")
async def order_endpoint(
    customer_id: str                  = Form(...,  description="رقم العميل"),
    message:     Optional[str]        = Form(None, description="نص الطلب"),
    audio:       Optional[UploadFile] = File(None, description="ملف صوتي"),
    language:    str                  = Form("ar", description="لغة الصوت: ar أو en"),
):
    """
    ┌─────────────────────────────────────────────┐
    │  نص فقط:                                    │
    │    customer_id = "123"                       │
    │    message     = "أريد بيتزا مارغريتا"      │
    │                                             │
    │  صوت فقط:                                   │
    │    customer_id = "123"                       │
    │    audio       = [ملف mp3/wav/m4a...]        │
    │                                             │
    │  الاثنين معاً:                              │
    │    customer_id = "123"                       │
    │    message     = "أريد بيتزا"               │
    │    audio       = [ملف صوتي]                 │
    └─────────────────────────────────────────────┘
    """

    # لازم يكون في صوت أو نص
    if not message and (not audio or not audio.filename):
        raise HTTPException(
            status_code=422,
            detail="أرسل نصاً (message) أو ملفاً صوتياً (audio) أو الاثنين."
        )

    audio_path    = None
    transcription = ""
    input_mode    = None

    try:
        # ── معالجة الصوت ─────────────────────────────────────────────────────
        if audio and audio.filename:
            ext = os.path.splitext(audio.filename)[1].lower()
            if ext not in ALLOWED_AUDIO:
                raise HTTPException(
                    status_code=400,
                    detail=f"نوع الملف '{ext}' غير مدعوم. المدعوم: {', '.join(ALLOWED_AUDIO)}"
                )
            content = await audio.read()
            if not content:
                raise HTTPException(status_code=400, detail="الملف الصوتي فارغ.")

            audio_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")
            with open(audio_path, "wb") as f:
                f.write(content)

            transcription = transcribe_audio(audio_path, language=language)
            if not transcription:
                raise HTTPException(status_code=422, detail="لم يتم التعرف على الصوت.")

        # ── تحديد النص النهائي للـ RAG ────────────────────────────────────────
        has_text  = bool(message and message.strip())
        has_voice = bool(transcription)

        if has_text and has_voice:
            input_mode = "both"
            final_text = f"{message.strip()} — {transcription}"
        elif has_voice:
            input_mode = "voice"
            final_text = transcription
        else:
            input_mode = "text"
            final_text = message.strip()

        # ── RAG chatbot ───────────────────────────────────────────────────────
        result = chat(final_text)

        # ── إرسال للـ API الخارجي ─────────────────────────────────────────────
        api_result = None
        if result["order_ready"] and result["order_data"]:

            if input_mode == "voice":
                api_result = send_voice_to_api(customer_id, audio_path)

            elif input_mode == "text":
                api_result = send_text_to_api(
                    customer_id, final_text, result["order_data"]
                )

            elif input_mode == "both":
                # يبعت الصوت + يضيف النص كـ field إضافي
                api_result = send_voice_to_api(customer_id, audio_path)
                api_result["text"] = final_text

        # ── الرد ─────────────────────────────────────────────────────────────
        response_body = {
            "input_mode":  input_mode,
            "reply":       result["reply"],
            "order_ready": result["order_ready"],
            "order_data":  result["order_data"],
            "api_result":  api_result,
        }
        if transcription:
            response_body["transcription"] = transcription

        return response_body

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
