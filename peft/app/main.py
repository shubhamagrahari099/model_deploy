print('shubham-1')
from fastapi import FastAPI
from pydantic import BaseModel
from app.inference import summarize
print('shubham-2')
app = FastAPI()

class DialogueInput(BaseModel):
    dialogue: str

@app.post("/summarize/")
def get_summary(dialogue_input: DialogueInput):
    summary = summarize(dialogue_input.dialogue)
    return {"summary": summary}
