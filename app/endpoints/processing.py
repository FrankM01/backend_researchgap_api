from fastapi import APIRouter, HTTPException, UploadFile, File
import re
import spacy
from collections import Counter,defaultdict
import json
import requests

from app.config import PDF_EXTRACTOR_API_KEY  

router = APIRouter()

# Configuración de la clave de API de OpenAI
API_KEY = PDF_EXTRACTOR_API_KEY  

router = APIRouter()
UPLOAD_DIR = 'uploads/'

BASE_URL = 'https://api.api2convert.com/v2'
nlp = spacy.load('en_core_web_sm') # Load NER Model


@router.post("/preprocess/")
async def preprocess_file(file: UploadFile = File(...)):
    try:
        print("Iniciando el proceso de preprocesamiento del archivo")
        job_id, server_url = create_job()
        if not job_id or not server_url:
            print("Error al crear el trabajo en la API")
            raise HTTPException(status_code=500, detail="Error al crear el trabajo en la API")

        print(f"Trabajo creado con éxito: {job_id}")
        upload_status = upload_file_to_job(server_url, job_id, file)
        if not upload_status:
            print("Error al subir el archivo PDF")
            raise HTTPException(status_code=500, detail="Error al subir el archivo PDF")

        print("Archivo subido con éxito")
        if not process_job(job_id):
            raise HTTPException(status_code=500, detail="Error al procesar el trabajo")

        text_content = get_converted_text(job_id)
        if not text_content:
            print("Error al obtener el texto convertido")
            raise HTTPException(status_code=500, detail="Error al obtener el texto convertido")

        print("Texto convertido obtenido exitosamente")
        print("Texto ",text_content)


        # texto_limpio = clean_text(text_content)
        secciones_extraidas = extract_sections(text_content)
        results = generate_json_with_ner(text_content, secciones_extraidas)

        json_file_name = f"{file.filename}_result.json"
        output_file = f"{UPLOAD_DIR}{json_file_name}"
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print("Archivo JSON guardado exitosamente")
        return {"message": "File processed successfully", "results": results, "output_file": json_file_name}
    except Exception as e:
        print("Error en el procesamiento del archivo:", e)
        raise HTTPException(status_code=500, detail="Error en el procesamiento del archivo")



def create_job():
    url = f"{BASE_URL}/jobs"
    headers = {'X-Oc-Api-Key': API_KEY, 'Content-Type': 'application/json'}
    data = {
        "type": "job",
        "process": False,
        "fail_on_input_error": True,
        "fail_on_conversion_error": True,
        "conversion": [{
            "category": "document",
            "target": "txt"
        }]
    }
    response = requests.post(url, headers=headers, json=data)
    
    print("Status Code:", response.status_code)
    print("Response Text:", response.text)

    if response.status_code == 201:
        job_info = response.json()
        return job_info.get('id'), job_info.get('server')
    return None, None

def upload_file_to_job(server_url, job_id, file):
    upload_url = f"{server_url}/upload-file/{job_id}"
    headers = {"X-Oc-Api-Key": API_KEY}
    files = {"file": (file.filename, file.file, file.content_type)}
    response = requests.post(upload_url, headers=headers, files=files)
    print("Status Code:", response.status_code)
    print("Response Text:", response.text)
    return response.status_code == 200

def process_job(job_id):
    url = f"{BASE_URL}/jobs/{job_id}"
    headers = {'X-Oc-Api-Key': API_KEY, 'Content-Type': 'application/json'}
    data = {"process": True}
    response = requests.patch(url, headers=headers, json=data)
    return response.status_code == 200

def get_converted_text(job_id):
    """Descarga el archivo de texto resultante de la conversión."""
    url = f"{BASE_URL}/jobs/{job_id}/output"
    headers = {"X-Oc-Api-Key": API_KEY}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        # Supongamos que el primer archivo de salida es el que queremos
        output_file_url = response.json()[0]["uri"]
        text_response = requests.get(output_file_url)
        if text_response.status_code == 200:
            return text_response.text
    return None


def clean_text(text):
    # Eliminar URLs y otros caracteres no deseados, pero preservamos saltos de línea
    text = re.sub(r'https?://\S+|doi:\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()  # No convertimos a minúsculas ni eliminamos saltos de línea

def apply_ner(text):
    doc = nlp(text)
    entities = {"authors": [], "institutions": [], "technologies": []}
    for ent in doc.ents:
        entity_text = re.sub(r"[\W_]+", " ", ent.text).strip().lower()
        if re.match(r"\b(?:https?|www)\b", entity_text) or entity_text.isdigit() or re.match(r"^[\d\s]+$", entity_text):
            continue
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
    section_pattern = r"(?:^|\n)\s*(\d+\s*(?:\.\d+\s*)?)?(?:[A-Z]\s*)*(results and discussions?|discussions?|conclusions?(?: and recommendations?)?|limitations?(?: of the study)?)\b[:\-\.\s]*\n?"

    # Encuentra coincidencias de secciones
    matches = list(re.finditer(section_pattern, text, flags=re.IGNORECASE))
    extracted_sections = {}

    for i, match in enumerate(matches):
        section_title = match.group(2).strip().lower()  # Captura solo el título de la sección

        # Determina el inicio y el fin del contenido de la sección
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        # Extrae el contenido de la sección
        content = text[start:end].strip()

        # Verifica si "REFERENCES" aparece en el contenido y corta el texto hasta antes de "REFERENCES"
        references_index = re.search(r"\n(?:[A-Z]\s*)*(references|R\s*E\s*F\s*E\s*R\s*E\s*N\s*C\s*E\s*S)\b", content, flags=re.IGNORECASE)
        if references_index:
            content = content[:references_index.start()].strip()

        # Agrega la sección al diccionario de secciones extraídas
        extracted_sections[section_title] = content

    print("Secciones extraídas:", extracted_sections)  # Imprime las secciones extraídas
    return extracted_sections

def generate_json_with_ner(text, sections):
    ner_data = apply_ner(text)
    result_json = {
        "authors": clean_authors(ner_data["authors"]),
        "institutions": clean_institutions(ner_data["institutions"]),
        "technologies": clean_technologies(ner_data["technologies"]),
        "sections": clean_sections(sections)
    }
    return result_json


# Funciones de limpieza

def clean_authors(authors):
    cleaned_authors = defaultdict(int)
    for author, count in authors.items():
        author = re.sub(r"[\W_]+", " ", author).strip().lower()
        if author not in ["fig", "table", "al", "page", "contributed", "applications", "et al"] and count > 1:
            cleaned_authors[author] += count
    return dict(cleaned_authors)

def clean_institutions(institutions):
    cleaned_institutions = defaultdict(int)
    for institution, count in institutions.items():
        institution = re.sub(r"[\W_]+", " ", institution).strip().lower()
        if institution not in ["fig", "page", "al", "doi"] and count > 1:
            cleaned_institutions[institution] += count
    return dict(cleaned_institutions)

def clean_technologies(technologies):
    cleaned_technologies = {}
    for technology, count in technologies.items():
        if not re.match(r"^\d+$", technology):
            technology = re.sub(r"[\W_]+", " ", technology).strip().lower()
            if technology not in ["by", "table", "page"]:
                cleaned_technologies[technology] = count
    return cleaned_technologies

def clean_sections(sections):
    cleaned_sections = {}
    for section, content in sections.items():
        content = re.sub(r"\[\d+\]", "", content)
        content = re.sub(r"https?://\S+|doi:\S+", "", content)
        content = re.sub(r"-\s+", "", content)
        content = re.sub(r"([a-zA-Z])([A-Z])", r"\1 \2", content)
        content = re.sub(r"\s+", " ", content).strip()
        cleaned_sections[section] = content
    return cleaned_sections