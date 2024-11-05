from fastapi import APIRouter, HTTPException, UploadFile, File
import pdfplumber
import re
import spacy
from collections import Counter,defaultdict
import json
import os

router = APIRouter()
UPLOAD_DIR = 'uploads/'
nlp = spacy.load('en_core_web_sm') # Load NER Model

@router.post("/preprocess/")
async def preprocess_file(file: UploadFile = File(...)):
    # *Save the uploaded file
    try:
        file_location = f"{UPLOAD_DIR}{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())
    except:
        raise HTTPException(status_code=500, detail= "Error saving file")
    

    # *Extract text from file
    try:
        with pdfplumber.open(file_location) as pdf:
            text = '\n\n'.join(extract_text_from_page(page) for page in pdf.pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error extracting text from file")

    # *Data cleaning and Apply NER
    try:
        texto_limpio = clean_text(text)
        secciones_extraidas = extract_sections(texto_limpio)
        results = generate_json_with_ner(texto_limpio, secciones_extraidas)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing data")

    
    # *Save JSON File
    json_file_name = f"{file.filename}_result.json"
    output_file = f"{UPLOAD_DIR}{json_file_name}"

    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    return {"message": "File processed successfully", "results": results, "output_file": json_file_name }



def extract_text_from_page(page):
    width_page = page.width
    height_page = page.height

    # Determinar si la pagina tiene dos columnas
    left_column = page.within_bbox((0, 0, width_page / 2, height_page))
    right_column = page.within_bbox((width_page / 2, 0, width_page, height_page))

    left_text = left_column.extract_text() or ""
    right_text = right_column.extract_text() or ""

    numbers_of_lines_left = len(left_text.split('\n'))
    numbers_of_lines_right = len(right_text.split('\n'))

    # Si tiene dos columnas extraer por separado
    if abs(numbers_of_lines_left - numbers_of_lines_right) < 10:
        return left_text + '\n\n' + right_text
    else:
        return page.extract_text() or ''


def clean_text(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower().strip()
    return text


def apply_ner(text):
    doc = nlp(text)
    entities = {"authors": [], "institutions": [], "technologies": []}
    for ent in doc.ents:
        entity_text = re.sub(r"[\W_]+", " ", ent.text).strip().lower()
        if ent.label_ == "PERSON" and entity_text not in ["fig", "table", "al", "page"]:
            entities["authors"].append(entity_text)
        elif ent.label_ == "ORG" and entity_text not in ["fig", "page", "al", "doi"]:
            entities["institutions"].append(entity_text)
        elif ent.label_ in ["TECHNOLOGY", "PRODUCT", "WORK_OF_ART", "STATE_OF_ART"]:
            entities["technologies"].append(entity_text)
    
    entities["authors"] = dict(Counter(entities["authors"]))
    entities["institutions"] = dict(Counter(entities["institutions"]))
    entities["technologies"] = dict(Counter(entities["technologies"]))
    return entities


def extract_sections(text):
    sections = re.split(r"(discussion\s*| limitation\s*| conclusion\s*)", text, flags=re.IGNORECASE)
    extracted_sections = {}
    for i in range(1, len(sections), 2):
        section_title = sections[i].strip().lower()
        content = sections[i + 1].strip()
        extracted_sections[section_title] = replace_unicode(content)
    
    return extracted_sections


def replace_unicode(text):
    replacements = {
        '\u2011': '-', '\u2013': '-', '\u2014': '-', '\u2018': "'", '\u2019': "'", '\u201C': '"', '\u201D': '"',
        '\u00e1': 'á', '\u00e9': 'é', '\u00ed': 'í', '\u00f3': 'ó', '\u00fa': 'ú', '\u00ef': 'ï', '\u00fc': 'ü', '\u00f1': 'ñ',
    }
    for unicode_char, replacement in replacements.items():
        text = text.replace(unicode_char, replacement)
    return text

def generate_json_with_ner(text, sections):
    ner_data = apply_ner(text)

    result_json = {
        "authors": clean_authors(ner_data["authors"]),
        "institutions": clean_institutions(ner_data["institutions"]),
        "technologies": clean_technologies(ner_data["technologies"]),
        "sections": clean_sections(sections)
    }

    return result_json


def clean_authors(authors):
    cleaned_authors = defaultdict(int)
    for author, count in authors.items():
        # Elimina caracteres no deseados y normaliza texto
        author = re.sub(r"[\W_]+", " ", author).strip().lower()
        if author not in ["fig", "table", "al", "page", "contributed", "applications"]:
            # Agrupa nombres similares
            if count > 1:
                cleaned_authors[author] += count
    return dict(cleaned_authors)


def clean_institutions(institutions):
    cleaned_institutions = defaultdict(int)
    for institution, count in institutions.items():
        institution = re.sub(r"[\W_]+", " ", institution).strip().lower()
        # Filtra URLs y nombres irrelevantes
        if institution not in ["fig", "page", "al", "doi"] and not re.match(r"https?|doi", institution):
            if count > 1:
                cleaned_institutions[institution] += count
    return dict(cleaned_institutions)


def clean_technologies(technologies):
    cleaned_technologies = {}
    for technology, count in technologies.items():
        # Elimina entradas que parecen fechas o números
        if not re.match(r"^\d+$", technology) and not re.match(r"^\d{4} \d{2} \d{2}$", technology):
            technology = re.sub(r"[\W_]+", " ", technology).strip().lower()
            # Excluye entradas ambiguas
            if technology not in ["by", "table", "page"]:
                cleaned_technologies[technology] = count
    return cleaned_technologies

def clean_sections(sections):
    cleaned_sections = {}
    for section, content in sections.items():
        content = re.sub(r"-\s+", "", content)
        content = re.sub(r"\s+", " ", content).strip()
        cleaned_sections[section] = content
    return cleaned_sections
            
