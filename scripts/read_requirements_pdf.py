
import sys
import os
# Install pypdf if not present (assuming standard environment, but will try-except)
try:
    from pypdf import PdfReader
except ImportError:
    # Fallback or error
    print("pypdf not installed. Please install it.")
    sys.exit(1)


def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for i, page in enumerate(reader.pages):
            try:
                text += page.extract_text() + "\n"
            except Exception as e:
                print(f"Error reading page {i}: {e}")
        return text
    except Exception as e:
        return f"Error opening PDF: {e}"

if __name__ == "__main__":
    pdf_path = r"C:\G\New folder (2)\QA Requirements.pdf"
    content = extract_text_from_pdf(pdf_path)
    
    with open("requirements_output.txt", "w", encoding="utf-8") as f:
        f.write(content)
    print("Successfully wrote to requirements_output.txt")

