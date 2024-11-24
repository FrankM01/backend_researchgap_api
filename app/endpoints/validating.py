import math
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import spacy
import logging

router = APIRouter()
nlp = spacy.load('en_core_web_sm')

DIVERSITY_THRESHOLD = 1.5

class ValidationRequest(BaseModel):
  research_gaps: Optional[str] = None
  authors: Optional[Dict[str, int]] = None
  institutions: Optional[Dict[str, int]] = None
  technologies: Optional[Dict[str, int]] = None
  sections: Optional[Dict[str, str]] = None


def calculate_shannon_entropy(data):
  total = sum(data.values())
  if total == 0:
    return 0
  
  entropy = 0
  for count in data.values():
    probability = count / total
    if probability > 0:
      entropy -= probability * math.log2(probability)
  return entropy

def apply_ner_to_section(text):
  doc = nlp(text)
  keywords = [ent.text.lower() for ent in doc.ents]
  return keywords

def calculate_keyword_frequencies(keywords):
  frequency = {}
  for keyword in keywords:
    frequency[keyword] = frequency.get(keyword, 0) + 1
  return frequency

@router.post("/validate/")
async def validate_entropy(request: ValidationRequest):
  try:
    print("Research Gaps Content:", request.research_gaps)
    if not request.sections or len(request.sections) == 0:
      raise HTTPException(status_code=400, detail="Sections data is missing or empty")
    if not any([request.authors, request.institutions, request.technologies]):
      raise HTTPException(status_code=400, detail="Metadata (authors, institutions, technologies) is missing or empty")
    
    logging.info(f"Validating entropy for authors, institutions and  technologies ")
    authors_entropy = calculate_shannon_entropy(request.authors)
    institutions_entropy = calculate_shannon_entropy(request.institutions)
    technologies_entropy = calculate_shannon_entropy(request.technologies)

    logging.info(f"Validating entropy for each section: {request.sections.keys()}")
    sections_entropies = {}
    for section_name, section_text in request.sections.items():
      keywords = apply_ner_to_section(section_text)
      keywords_frequency = calculate_keyword_frequencies(keywords)
      sections_entropy = calculate_shannon_entropy(keywords_frequency)
      sections_entropies[section_name] = sections_entropy
    
    research_gap_entropy = 0
    if request.research_gaps:
      logging.info("Validating entropy for research gaps")
      research_gap_keywords = apply_ner_to_section(request.research_gaps) if request.research_gaps else []
      print(f"Keywords extracted from research gaps: {research_gap_keywords}")
      research_gap_frequency = calculate_keyword_frequencies(research_gap_keywords)
      if not research_gap_frequency:
        print("No keywords found in research gaps")
      else:
        print("Frequencies of Research Gaps Keywords:", research_gap_frequency)
      research_gap_entropy = calculate_shannon_entropy(research_gap_frequency)
      print("Research Gap Entropy:", research_gap_entropy)
    

    # Evaluar diversidad
    diversity_evaluation = {
      "authors": "Alta diversidad" if authors_entropy > DIVERSITY_THRESHOLD else "Baja diversidad",
      "institutions": "Alta diversidad" if institutions_entropy > DIVERSITY_THRESHOLD else "Baja diversidad",
      "technologies": "Alta diversidad" if technologies_entropy > DIVERSITY_THRESHOLD else "Baja diversidad",
      "sections": {
        section: "Alta diversidad" if entropy > DIVERSITY_THRESHOLD else "Baja diversidad"
        for section, entropy in sections_entropies.items()
      },
      "research_gaps": "Alta diversidad" if research_gap_entropy > DIVERSITY_THRESHOLD else "Baja diversidad"
    }

    return {
        "message": "Validation completed successfully",
        "entropy_values": {
            "authors_entropy": authors_entropy,
            "institutions_entropy": institutions_entropy,
            "technologies_entropy": technologies_entropy,
            **sections_entropies,
            "research_gap_entropy": research_gap_entropy
        },
        "diversity_evaluation": diversity_evaluation,
        "research_gaps": request.research_gaps
    }
  except Exception as e:
    print("Error en la validación de la entropía:", e)
    raise HTTPException(status_code=500, detail="Error en la validación de la entropía")


