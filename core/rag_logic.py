import chromadb
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def get_document_text(uploaded_file):
    text = ""
    file_name = uploaded_file.name
    file_extension = file_name.split('.')[-1].lower()

    if file_extension == 'pdf':
        pdf_reader = PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file_extension == 'docx':
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file_extension == 'txt':
        text = uploaded_file.read().decode("utf-8")
    else:
        return None
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def create_vector_store(text_chunks):
    print("Создание эмбеддингов для фрагментов текста")
    embeddings = embedding_model.encode(text_chunks, show_progress_bar=True)
    print("Эмбеддинги созданы")

    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="document_collection")

    chunk_ids = [str(i) for i in range(len(text_chunks))]

    collection.add(
        embeddings=embeddings.tolist(),
        documents=text_chunks,
        ids=chunk_ids
    )
    print("Векторная база создана и готова")

    return collection


def search_in_vector_store(collection, query):
    query_embedding = embedding_model.encode(query)  # Убрал show_progress_bar, для одного запроса он не нужен

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )

    return results['documents'][0]