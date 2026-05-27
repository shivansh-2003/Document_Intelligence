from services.parsing.pdf_parser import extract_full_text,extract_text_pages
from services.parsing.doc_parser import extract_full_text as extract_docx_text, extract_text_pages as extract_docx_pages
from services.parsing.pptx_parser import extract_full_text as extract_pptx_text, extract_text_slides as extract_pptx_slides
pdf_path ="/Users/shivanshmahajan/Developer/Docs/AI_engineering.pdf" # Note the leading /
docx_path = "test_files/agentic.docx"
pptx_path = "test_files/CRAG.pptx"

# pages = extract_text_pages(pdf_path, print_pages=True)
# pages=extract_docx_pages
pages=extract_pptx_slides(pptx_path)
print(pages)