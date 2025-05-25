from fastapi import FastAPI, Request
from pydantic import BaseModel
from langdetect import detect
from app.chatbot import get_response
from app.translator import translate
from app.scheduler import init_db, schedule_appointment

app = FastAPI()
init_db()

class Message(BaseModel):
    message: str

class Appointment(BaseModel):
    name: str
    date: str
    time: str
    doctor: str

@app.post("/chat")
async def chat(msg: Message):
    user_input = msg.message
    detected_lang = detect(user_input)

    # Translate to English if necessary
    if detected_lang != "en":
        user_input = translate(user_input, detected_lang, "en")

    # Get response from chatbot
    response = get_response(user_input)

    # Translate back to user's language
    if detected_lang != "en":
        response = translate(response, "en", detected_lang)

    return {"response": response}

@app.post("/appointment")
async def appointment(app_data: Appointment):
    schedule_appointment(app_data.name, app_data.date, app_data.time, app_data.doctor)
    return {"status": "Appointment scheduled successfully!"}
