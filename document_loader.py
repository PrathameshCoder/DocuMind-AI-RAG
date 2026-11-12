
from pypdf import PdfReader
from io import BytesIO


def extract_text(uploaded_file):

    # Identify the file type using its extension
    filename = uploaded_file.name.lower()

    # Case 1: TXT file
    if filename.endswith(".txt"):
        text = uploaded_file.getvalue().decode("utf-8-sig")
        return text

    # Case 2: PDF file
    elif filename.endswith(".pdf"):

        # Convert uploaded bytes into a file-like object
        pdf_bytes = BytesIO(uploaded_file.getvalue())

        # Open PDF using pypdf
        reader = PdfReader(pdf_bytes)

        text = ""

        # Extract text from every page
        for page_number, page in enumerate(reader.pages):

            page_text = page.extract_text() or ""

            text += (
                f"\n\n--- Page {page_number + 1} ---\n"
                + page_text
            )

        return text

    else:
        raise ValueError("Unsupported file type")
