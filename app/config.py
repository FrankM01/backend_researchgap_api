import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PDF_EXTRACTOR_API_KEY = os.getenv("PDF_EXTRACTOR_API_KEY")