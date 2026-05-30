from fastapi import FastAPI
from pydantic import BaseModel

from backend.models.classifier import MentalHealthClassifier


app = FastAPI(
    title="Mental Health Agentic AI Platform"
)

classifier = MentalHealthClassifier()


class UserInput(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "Mental Health Agentic AI Platform Running"
    }


@app.post("/classify")
def classify_text(user_input: UserInput):
    prediction = classifier.predict(user_input.text)

    return prediction