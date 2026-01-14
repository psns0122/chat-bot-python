import os
from dotenv import load_dotenv

load_dotenv(".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY가 .env에 없습니다.")