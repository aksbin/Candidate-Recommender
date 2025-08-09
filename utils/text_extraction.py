import pdfplumber

# Extract plain text from a PDF upload; raise ValueError for non-PDF.
def extract_text(uploaded_file) -> str:
 
    # Require PDFs for consistent parsing
    if uploaded_file.type != "application/pdf":
        raise ValueError(f"Unsupported file type: {uploaded_file.type}. Only PDFs are supported.")

    text = []
    
    # Open PDF and collect page text
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)

    # Join pages with newlines
    return "\n".join(text)
