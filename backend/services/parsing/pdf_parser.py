from PyPDF2 import PdfReader


def extract_full_text(file_path: str) -> str:
    """Extract all text from a PDF as a single string."""
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def extract_text_pages(file_path: str, print_pages: bool = False) -> list[dict]:
    """
    Extract text from a PDF page by page.
    
    Returns a list of dicts: [{"page": 1, "text": "..."}, ...]
    Optionally prints each page with clear formatting.
    """
    reader = PdfReader(file_path)
    pages = []
    
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        page_data = {"page": i, "text": text.strip()}
        pages.append(page_data)
        
    
    return pages

