from fastapi import APIRouter, HTTPException
import openai
from app.config import OPENAI_API_KEY  # Configuración de la clave de OpenAI

router = APIRouter()

# Configuración de la clave de API de OpenAI
openai.api_key = OPENAI_API_KEY 

@router.post("/analyze/")
async def analyze(data: dict):
    try:
        # Extract sections
        sections = data.get("sections", {})
        discussion = sections.get("discussion", "")
        conclusion = sections.get("conclusion", "")
        limitation = sections.get("limitation", "")

        if not (discussion or conclusion or limitation):
            raise HTTPException(status_code=400, detail="No relevant sections found in the data.")

        # Build a conditional prompt
        message_content = "Below are sections from a scientific article. Identify and describe any research gaps present in the text."
        if discussion:
            message_content += f"\n\nDiscussion:\n{discussion}"
        if conclusion:
            message_content += f"\n\nConclusion:\n{conclusion}"
        if limitation:
            message_content += f"\n\nLimitation:\n{limitation}"
        
        message_content += "\n\nPlease provide the main research gaps identified in the text."

        messages = [
            {"role": "system", "content": "You are an AI specialized in identifying research gaps in scientific articles."},
            {"role": "user", "content" : message_content}
        ]

        # Call OpenAI model
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )

        research_gaps = response.choices[0].message["content"].strip()
        
        return {
            "message": "Research gaps identified successfully",
            "research_gaps": research_gaps
        }

    except Exception as e:
        print("Other error:", e)
        raise HTTPException(status_code=500, detail="Unknown error occurred in analysis.")
