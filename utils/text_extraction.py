import pdfplumber


def extract_text(uploaded_file) -> str:
    """
    Extract all text from a PDF file using pdfplumber.
    Raises:
        ValueError: if the uploaded file is not a PDF.
    Returns:
        A single string containing the extracted text.
    """
    # Ensure correct MIME type for PDF
    if uploaded_file.type != "application/pdf":
        raise ValueError(f"Unsupported file type: {uploaded_file.type}. Only PDFs are supported.")

    text = []
    # Open the uploaded file as a PDF
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)

    # Join all page texts with newlines
    return "\n".join(text)
