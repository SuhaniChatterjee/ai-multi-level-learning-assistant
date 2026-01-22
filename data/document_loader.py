from pypdf import PdfReader


def load_pdf(file_path: str) -> str:
    """
    Loads a PDF file and returns all extracted text as a single string.
    """
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text