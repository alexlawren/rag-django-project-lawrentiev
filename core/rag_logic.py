import pypdf
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def get_document_text(uploaded_file):
    """
    Извлекает текст из загруженного файла (.pdf, .docx, .txt).
    """
    text = ""
    file_name = uploaded_file.name
    file_extension = file_name.split('.')[-1]

    if file_extension == 'pdf':
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file_extension == 'docx':
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "/n"
    elif file_extension == 'txt':
        text += uploaded_file.read().decode("utf-8")
    else :
        return None
    return text

def get_text_chunks(text):
    """
    Разбивает большой текст на меньшие фрагменты (чанки).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks