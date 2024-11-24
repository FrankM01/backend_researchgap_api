from fastapi import APIRouter, HTTPException
import openai
from app.config import OPENAI_API_KEY  # Configuración de la clave de OpenAI

router = APIRouter()

# Configuración de la clave de API de OpenAI
openai.api_key = OPENAI_API_KEY 

SECTION_MAPPING = {
    "results and discussions": "discussion",
    "discussion": "discussion",
    "discussions": "discussion",
    "conclusions and recommendations": "conclusion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "limitations of the study": "limitation",
    "limitation": "limitation",
    "limitations": "limitation"
}


@router.post("/analyze/")
async def analyze(data: dict):
    try:
        # Extract sections
        sections = data.get("sections", {})
        normalized_sections = {
            SECTION_MAPPING.get(key.lower(), key) : value
            for key, value in sections.items()
            if key.lower() in SECTION_MAPPING
        }

        discussion = normalized_sections.get("discussion", "")
        conclusion = normalized_sections.get("conclusion", "")
        limitation = normalized_sections.get("limitation", "")

        authors = data.get("authors", {})
        institutions = data.get("institutions", {})
        technologies = data.get("technologies", {})

        if not (discussion or conclusion or limitation):
            raise HTTPException(status_code=400, detail="No relevant sections found in the data.")

        # Build a conditional prompt
        message_content = "Below are sections and metadata from a scientific article. Identify and describe any research gaps present in the text."

        if authors:
            authors_list = ", ".join([f"{author} ({count})" for author, count in authors.items()])
            message_content += f"\n\nAuthors (with frequency):\n{authors_list}"
        if institutions:
            institutions_list = ", ".join([f"{institution} ({count})" for institution, count in institutions.items()])
            message_content += f"\n\nInstitutions (with frequency):\n{institutions_list}"
        if technologies:
            technologies_list = ", ".join([f"{technology} ({count})" for technology, count in technologies.items()])
            message_content += f"\n\nTechnologies:\n{technologies_list}"
        if discussion:
            message_content += f"\n\nDiscussion:\n{discussion}"
        if conclusion:
            message_content += f"\n\nConclusion:\n{conclusion}"
        if limitation:
            message_content += f"\n\nLimitation:\n{limitation}"
        
        message_content += "\n\nPlease provide the main research gaps identified in the text (only one research gap). "

        messages = [
            {"role": "system", "content": "You are an AI specialized in identifying research gaps in scientific articles."},
            {"role": "user", "content" : message_content}
        ]

        # Call OpenAI model
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=200,
            temperature=0.7
        )

        research_gaps = response.choices[0].message["content"].strip()
        print('Research gaps:', research_gaps)
        
        return {
            "message": "Research gaps identified successfully",
            "research_gaps": research_gaps,
            "authors": authors,
            "institutions": institutions,
            "technologies": technologies
        }

    except Exception as e:
        print("Other error:", e)
        raise HTTPException(status_code=500, detail="Unknown error occurred in analysis.")
