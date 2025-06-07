import pdfplumber
import os
import sys

# Ensure the user provided a PDF path
if len(sys.argv) < 2:
    print("Usage: python extract_pdf.py <path_to_pdf>")
    sys.exit(1)

# Get the PDF path from the first command-line argument
pdf_path = sys.argv[1]

# Validate the file
if not os.path.isfile(pdf_path):
    print(f"Error: File not found: {pdf_path}")
    sys.exit(1)

# Generate output .txt file name
base_name = os.path.splitext(pdf_path)[0]
txt_path = base_name + ".txt"

# Open and extract text
with pdfplumber.open(pdf_path) as pdf:
    full_text = ""
    page_num = 0
    for page in pdf.pages:
        page_num += 1
        full_text += f"\n~~~Begin Page {page_num}~~~\n"
        text = page.extract_text()
        if text:
            full_text += text + "\n"
        full_text += f"~~~End Page {page_num}~~~\n"

# Write to .txt file
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"Extracted text saved to {txt_path}")
