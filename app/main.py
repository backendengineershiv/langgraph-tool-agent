from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from app.graph import graph

app = FastAPI(title="Simple LangGraph Agent")


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.thread_id}}

    result = graph.invoke(
        {"messages": [HumanMessage(content=req.message)]},
        config=config,
    )

    reply = result["messages"][-1].content
    return ChatResponse(reply=reply)


@app.get("/")
def health():
    return {"status": "ok"}
