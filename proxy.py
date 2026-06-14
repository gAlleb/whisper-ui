import os
import httpx
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List
from google import genai
from google.genai import types

app = FastAPI()

# Конфигурация
WHISPER_URL = "http://whisper:9000/v1/audio/transcriptions"
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.5-flash"  # Используем 3.5-flash

# Инициализация официального клиента Google Gen AI
client = None
if GEMINI_KEY:
    client = genai.Client(api_key=GEMINI_KEY)
else:
    print("ВНИМАНИЕ: GEMINI_API_KEY не установлен. Доступен только сырой Whisper.")

# Схемы данных для чата
class ChatMessage(BaseModel):
    role: str  # 'user' или 'model'
    text: str

class ChatRequest(BaseModel):
    context: str
    message: str
    history: List[ChatMessage] = []


# --- ЭНДПОИНТ 1: Распознавание и жесткая чистка (БЕЗ отсебятины) ---
@app.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    language: str = Form("ru"),
    clean_with_gemini: bool = Form(False),
    context: str = Form("")
):
    # 1. Локальный Whisper
    async with httpx.AsyncClient(timeout=1800.0) as httpx_client:
        file_content = await file.read()
        files = {"file": (file.filename, file_content, file.content_type)}
        data = {"model": "whisper-1", "language": language}
        
        try:
            response = await httpx_client.post(WHISPER_URL, files=files, data=data)
            response.raise_for_status()
            whisper_text = response.json().get("text", "")
        except Exception as e:
            return {"text": f"Ошибка Whisper:\n{str(e)}", "raw": True}

    if not clean_with_gemini or not client:
        return {"text": whisper_text, "raw": True}

    # Жесткие инструкции против галлюцинаций и отсебятины
    system_instruction = (
        "Ты — строгий редактор-корректор. Твоя единственная задача — исправить грамматику, "
        "расставить знаки препинания и разбить сырой текст распознавания речи (STT) на логические абзацы.\n"
        "ЖЕСТКИЕ ПРАВИЛА:\n"
        "1. НИ В КОЕМ СЛУЧАЕ не добавляй никакой информации от себя.\n"
        "2. НЕ придумывай факты, имена, цифры, даты или выводы, которых не было в исходном тексте.\n"
        "3. Текст должен оставаться максимально близким к оригиналу. Убирай только слова-паразиты (э-э, ну, как бы), "
        "повторы и заикания.\n"
        "4. Если фраза кажется бессмысленной, оставь её как есть, просто исправив орфографию. Не пытайся переписать её смысл."
    )
    
    user_prompt = f"Контекст разговора: {context}\n\nСырой текст STT:\n{whisper_text}"
    
    # Устанавливаем температуру в 0.0 для максимальной точности
    config = types.GenerateContentConfig(
        system_instruction=system_instruction, 
        temperature=0.0
    )

    try:
        # Используем АСИНХРОННЫЙ вызов client.aio (решает проблему зависания)
        gemini_response = await client.aio.models.generate_content(
            model=MODEL_NAME, 
            contents=user_prompt, 
            config=config
        )
        return {"text": gemini_response.text, "raw": False}
    except Exception as e:
        return {"text": f"⚠️ Ошибка Gemini: {str(e)}\n\n{whisper_text}", "raw": True}


# --- ЭНДПОИНТ 2: Асинхронный чат по тексту расшифровки ---
@app.post("/chat")
async def chat_with_transcript(req: ChatRequest):
    if not client:
        return {"error": "API-ключ Gemini не настроен"}
    
    prompt = (
        "Ты — умный ИИ-ассистент. Перед тобой текст расшифровки аудиозаписи.\n"
        "Твоя задача — отвечать на вопросы пользователя, используя исключительно этот текст.\n"
        "Отвечай четко, по делу, структурируй списки, если это необходимо.\n"
        "Если в тексте нет ответа на вопрос, вежливо скажи, что в записи об этом не говорилось.\n\n"
        f"=== НАЧАЛО ТЕКСТА РАСШИФРОВКИ ===\n{req.context}\n=== КОНЕЦ ТЕКСТА РАСШИФРОВКИ ===\n\n"
    )
    
    if req.history:
        prompt += "История вашего текущего диалога:\n"
        for msg in req.history:
            prefix = "Пользователь" if msg.role == "user" else "Ассистент"
            prompt += f"{prefix}: {msg.text}\n"
        prompt += "\n"
        
    prompt += f"Пользователь: {req.message}\n"
    prompt += "Ассистент:"

    try:
        # Используем АСИНХРОННЫЙ вызов client.aio (решает проблему зависания)
        response = await client.aio.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return {"text": response.text}
    except Exception as e:
        return {"error": f"Ошибка генерации: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
