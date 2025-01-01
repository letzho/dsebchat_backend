from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_text_splitter(chunk_size=1000, chunk_overlap=200):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    ) 