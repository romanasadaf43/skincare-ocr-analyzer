from flask import Flask, render_template, request
from flask_cors import CORS
import json
import os
from PIL import Image
import pytesseract
from rapidfuzz import fuzz
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import re

# Flask setup
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, "..", "templates")
static_dir = os.path.join(base_dir, "..", "static")
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

CORS(app)
print("Template Path:", os.path.abspath(app.template_folder))
print("Static Path:", os.path.abspath(app.static_folder))

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load product data
json_path = os.path.join(base_dir, '../products.json')

# Helper functions to safely parse numbers
def safe_int(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def normalize_product(p):
    return {
        "name": p.get("Product Name", "").strip(),
        "brand": p.get("Brand", "").strip(),
        "ingredient": p.get("Key Ingredient", "").strip(),
        "risk_score": safe_int(p.get("Risk Score", 0)),
        "effectiveness_score": safe_float(p.get("Effectiveness Score (Based on Key Ingredient)", 0))
    }

with open(json_path) as f:
    raw_products = json.load(f)
    products = [normalize_product(p) for p in raw_products]

# NLP models
nlp = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

known_brands = [
    "dr sheth", "dot and key", "minimalist", "plum", "neutrogena", "mamaearth",
    "pond", "wow", "the ordinary", "cetaphil", "nivea", "biotique", "himalaya",
    "forest essentials", "lotus", "lotus professional"
]

BRAND_CORRECTIONS = {
    "mamaearth": "mamaearth", "mameaearth": "mamaearth", "mameearth": "mamaearth",
    "plem": "plum", "pond?": "pond",
    "dr sheths": "dr. sheth's", "dr sheth": "dr. sheth's", "dr. sheths": "dr. sheth's", "dr. sheth": "dr. sheth's",
    "l@tus": "lotus", "latus": "lotus", "lotus prof": "lotus professional"
}

def correct_text(text):
    text = text.replace("&", "and").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    for wrong, correct in BRAND_CORRECTIONS.items():
        text = re.sub(rf'\b{re.escape(wrong)}\b', correct, text, flags=re.IGNORECASE)
    return text

def extract_text_from_image(image_path):
    img = Image.open(image_path)
    raw_text = pytesseract.image_to_string(img)
    return correct_text(raw_text)

def extract_entities(text):
    ner_entities = nlp(text)
    ner_words = [e["word"].replace("##", "") for e in ner_entities if e["score"] > 0.85]

    keyword_candidates = re.findall(r'\b\w+\b', text.lower())
    important_words = [
        w for w in keyword_candidates 
        if len(w) > 3 and w not in {"with", "face", "wash", "and", "the"}
    ]

    fuzzy_brand_matches = []
    for brand in known_brands:
        score = fuzz.partial_ratio(brand, text.lower())
        if score > 80:
            fuzzy_brand_matches.append(brand)

    combined = ner_words + important_words + fuzzy_brand_matches + [text.replace("\n", " ")]
    normalized_entities = [e.lower().strip() for e in set(combined) if len(e.strip()) > 2]

    return normalized_entities

def match_product(entities, products):
    results = []
    all_text = correct_text(" ".join(entities)).lower()

    for product in products:
        brand = correct_text(product.get("brand", "").lower())
        name = product.get("name", "").lower()
        ingredient = product.get("ingredient", "").lower()

        best_score = 0
        for entity in entities:
            e = entity.lower()
            name_score = fuzz.token_set_ratio(e, name)
            brand_score = fuzz.partial_ratio(e, brand)
            ingredient_score = fuzz.token_set_ratio(e, ingredient)

            total_score = (
                0.5 * name_score +
                0.3 * ingredient_score +
                0.2 * brand_score
            )
            best_score = max(best_score, total_score)

        boost = 0
        if brand in all_text:
            boost += 10
        if ingredient in all_text:
            boost += 5
        name_words = name.split()
        if sum(1 for word in name_words if word in all_text) >= 2:
            boost += 5

        boosted_score = best_score + boost
        results.append((product, boosted_score))

    results.sort(key=lambda x: x[1], reverse=True)
    top_results = results[:5]

    print("\n[CANDIDATES]")
    for prod, score in top_results:
        print(f"- {prod['brand']} - {prod['name']} => Score: {score}")

    if not top_results or top_results[0][1] < 50:
        return None, 0

    if entities:
        ocr_query = " ".join(entities)
        query_embed = sbert_model.encode(ocr_query, convert_to_tensor=True)
        candidate_names = [r[0]["name"] for r in top_results]
        product_embeds = sbert_model.encode(candidate_names, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embed, product_embeds)[0]
        best_idx = int(scores.argmax())
        return top_results[best_idx]

    return top_results[0]

@app.route("/", methods=["GET", "POST"])
def index():
    matched_product = None
    match_score = None
    message = None

    if request.method == "POST":
        image = request.files["image"]
        if image:
            image_path = os.path.join(UPLOAD_FOLDER, image.filename)
            image.save(image_path)

            text = extract_text_from_image(image_path)
            print("\n[OCR TEXT]:", text)

            entities = extract_entities(text)
            print("\n[EXTRACTED ENTITIES]:", entities)

            matched_product, match_score = match_product(entities, products)
            if not matched_product:
                message = "No product matched. Try uploading a clearer image or adjusting lighting."

    return render_template(
        "index.html",
        product=matched_product,
        score=match_score,
        message=message
    )

if __name__ == "__main__":
    app.run(debug=True)
