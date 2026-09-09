from fastapi import FastAPI
from services.llm import llm

app = FastAPI()


@app.get("/")
def home():
    response = llm.invoke("Say hello in one sentence.")
    return {"message": response.content}