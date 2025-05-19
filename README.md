# Skincare OCR Analyzer

This is a web application that uses OCR (Optical Character Recognition) to extract ingredients from skincare product labels and analyze them for safety, effectiveness, and skin type compatibility.

---

## 🚀 Features

- 📸 Upload images of skincare product labels
- 🤖 Automatic OCR and text cleaning
- 🔍 Ingredient extraction using keyword and fuzzy matching
- 🧪 Analysis using a dataset of 500 Indian skincare products
- 📊 Displays safety, effectiveness, skin type suitability, and more

---

## 🛠 Tech Stack

- **Backend:** Python (Flask)
- **Frontend:** HTML, JavaScript
- **OCR:** Tesseract via `pytesseract`
- **Matching:** FuzzyWuzzy
- **Data:** `products.json` (converted from CSV)

---

## 📁 Project Structure

skincare-ocr-analyzer/
│
├── app.py # Main Flask backend
├── products.json # Ingredient database
├── requirements.txt # Python dependencies
├── templates/
│ └── index.html # Frontend HTML
├── static/
│ ├── uploads/ # Uploaded label images
│ └── style.css # (Optional) Styling
└── README.md # You're here!