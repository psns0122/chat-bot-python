from pydantic import BaseModel

class ChatModel(BaseModel):
    sessionId: str
    message: str