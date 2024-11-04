from fastapi import APIRouter, HTTPException, UploadFile, File
import pdfplumber
import re
import spacy
from collections import Counter
import json
import os

router = APIRouter()
UPLOAD_DIR = 'uploads/'
nlp = spacy.load('en_core_web_sm') # Load NER Model

@router.post("/preprocess/")
async def preprocess_file(file: UploadFile = File(...)):
    # Save the uploaded file
    try:
        file_location = f"{UPLOAD_DIR}{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())
    except:
        raise HTTPException(status_code=500, detail= "Error saving file")
    

    # Extract text from file
    try:
        with pdfplumber.open(file_location) as pdf:
            text = '\n\n'.join(extract_text_from_page(page) for page in pdf.pages)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error extracting text from file")

    # Remove special characters (Data cleaning) and Apply NER
    try:
        texto_limpio = clean_text(text)
        entidades_completas = apply_ner(texto_limpio)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing data")

    # Filtrar entidades
    authors = [entity for entity, label in entidades_completas if label == 'PERSON']
    institutions = [entity for entity, label in entidades_completas if label == 'ORG']
    technologies = [entity for entity, label in entidades_completas if label in ['TECHNOLOGY', 'PRODUCT', 'WORK_OF_ART','STATE_OF_ART']]

    # Contar entidades e identificar tecnologías emergentes
    authors_freq = Counter(authors)
    institutions_freq = Counter(institutions)
    technologies_freq = Counter(technologies)
    emergents_technologies = [technology for technology, freq in technologies_freq.items() if freq == 1]

    # Extraer secciones clave
    secciones_extraidas = extract_sections(texto_limpio)

    # Decodificar caracteres Unicode
    for section, content in secciones_extraidas.items():
        secciones_extraidas[section] = replace_unicode(content)

    # Generar resultados en formato JSON
    results = {
        'authors': dict(authors_freq),
        'institutions': dict(institutions_freq),
        'technologies': dict(technologies_freq),
        'emergent_technologies': emergents_technologies,
        'sections': secciones_extraidas
    }

    output_file = f"{UPLOAD_DIR}{file.filename}_result.json"
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    return {"message": "File processed successfully", "results": results}



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
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities


def extract_sections(text):
    sections = re.split(r"(abstract\s*| discussion\s*| limitation\s*| conclusion\s*)", text, flags=re.IGNORECASE)
    extracted_sections = {}
    for i in range(1, len(sections), 2):
        section_title = sections[i].strip().lower()
        extracted_sections[section_title] = sections[i + 1].strip()
    return extracted_sections


def replace_unicode(text):
    replacements = {
        '\u2011': '-',  # Non-breaking hyphen
        '\u2013': '-',  # En dash
        '\u2014': '-',  # Em dash
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u201C': '"',  # Left double quote
        '\u201D': '"',  # Right double quote
        '\u00e1': 'á',  # a with acute
        '\u00e9': 'é',  # e with acute
        '\u00ed': 'í',  # i with acute
        '\u00f3': 'ó',  # o with acute
        '\u00fa': 'ú',  # u with acute
        '\u00ef': 'ï',  # i with diaeresis
        '\u00fc': 'ü',  # u with diaeresis
        '\u00f1': 'ñ',  # n with tilde
    }

    for unicode_char, replacement in replacements.items():
        text = text.replace(unicode_char, replacement)
    return text
