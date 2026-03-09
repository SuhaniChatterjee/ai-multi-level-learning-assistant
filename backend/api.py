from fastapi import FastAPI
from pydantic import BaseModel

from backend.rag_service import answer_question

app = FastAPI()


class QuestionRequest(BaseModel):
    question: str
    level: str


@app.post("/ask")
def ask_question(req: QuestionRequest):

    answer = answer_question(req.question, req.level)

    return {"answer": answer}