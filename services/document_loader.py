from langchain.document_loaders import UnstructuredFileLoader, DocxLoader, WebBaseLoader
from fastapi import UploadFile
import os

async def load_document(source, is_url=False):
    try:
        if is_url:
            # Load from URL
            loader = WebBaseLoader(source)
            documents = loader.load()
        else:
            # Load from file
            if isinstance(source, UploadFile):
                # Save uploaded file temporarily
                temp_path = f"temp_{source.filename}"
                with open(temp_path, "wb") as temp_file:
                    content = await source.read()
                    temp_file.write(content)
                
                if source.filename.endswith(".docx"):
                    loader = DocxLoader(temp_path)
                else:
                    loader = UnstructuredFileLoader(temp_path)
                
                documents = loader.load()
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            else:
                # Load local file
                if source.endswith(".docx"):
                    loader = DocxLoader(source)
                else:
                    loader = UnstructuredFileLoader(source)
                documents = loader.load()
                
        return documents
    except Exception as e:
        print(f"Error loading document: {e}")
        raise e 