from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from pydantic import BaseModel
from typing import List, Optional
from functools import lru_cache
import os
import re
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import WebBaseLoader, Docx2txtLoader,UnstructuredURLLoader,SeleniumURLLoader,PlaywrightURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from datetime import datetime,timedelta
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate,HumanMessagePromptTemplate
# from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import Pinecone,ServerlessSpec
from typing import List, Dict
import uuid
import time
import pinecone




load_dotenv(find_dotenv(), override=True)
api_key = os.environ.get("PINECONE_API_KEY")
pc=Pinecone(api_key=api_key)
vector_store=None
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://dsebchat-frontend-29da6c2ca4ad.herokuapp.com/",  # Add this line too for completeness
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PERSIST_DIRECTORY = "./chroma_db"
NYP_URL = "https://www.nyp.edu.sg/student/study/schools/engineering/diploma-sustainability-engineering-business"
certificate_URL = "https://www.nyp.edu.sg/student/study/schools/engineering/stories/engineering-students-to-graduate-with-additional-industry-recognised-certifications"
scholarship_URL = "https://www.nyp.edu.sg/student/study/scholarships-financial-matters/scholarships-study-awards"
coursefee_URL="https://www.nyp.edu.sg/student/study/scholarships-financial-matters/fees/annual-course-fees"
finance_scheme_URL="https://www.nyp.edu.sg/student/study/scholarships-financial-matters/schemes"
financial_aid_URL="https://www.nyp.edu.sg/student/study/scholarships-financial-matters/financial-assistance/bursaries"
admission_URL="https://www.nyp.edu.sg/student/study/admissions"


current_dir = os.path.dirname(os.path.abspath(__file__))
QNA_PATH = os.path.join(current_dir, "..", "QnA_03.docx")

class QueryRequest(BaseModel):
    query: str
    collection_name: Optional[str] = "default"



def create_vector_store():
    global all_docs
    try:
        # List of URLs to load
        urls = [
            NYP_URL,
            # certificate_URL,
            # scholarship_URL,
            # coursefee_URL,
            # finance_scheme_URL,
            # financial_aid_URL,
            # admission_URL
        ]
        
        # Load all URLs
        all_docs = []
        
    
        # Load DOCX content if exists
        if os.path.exists(QNA_PATH):
            docx_loader = Docx2txtLoader(QNA_PATH)
            docx_docs = docx_loader.load()
            for doc in docx_docs:
                doc.page_content=clean_document_content(doc.page_content)
                # Split content into Year 1 and Year 2 segments dynamically
                # segmented_docs = segment_by_years(doc.page_content)
                all_docs.append(doc) 

                
            print(f"Loaded content from: {QNA_PATH}")
        else:
            print(f"Warning: {QNA_PATH} not found")
        
        for url in urls:
            try:
                web_loader = WebBaseLoader(url)
                
                docs = web_loader.load()
                for doc in docs:
                    doc.page_content=clean_document_content(doc.page_content)
                    all_docs.append(doc)
                if docs:
                    all_docs.append(doc)
                    print(f"Loaded content from: {url} - {len(all_docs)} documents")
                else:
                    print(f"No documents found at: {url}")
                
                
            except Exception as e:
                print(f"Error loading {url}: {str(e)}")
                continue

        if not all_docs:
            raise Exception("No documents were loaded successfully")

        print(f"Total documents loaded: {len(all_docs)}")
        

        return all_docs
    
    except Exception as e:
        print(f"Error creating vector store: {e}")
        raise

def clean_document_content(text: str) -> str:
    # Remove excessive newlines
    text = re.sub(r'\n+', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_data(data,chunk_size=1024):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=400,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(data)
    
    return chunks



def create_embeddings_pinecone(chunks,index_name="nyp-chatbot"):
    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    from pinecone import PodSpec,ServerlessSpec
    
    
    embeddings=OpenAIEmbeddings(model='text-embedding-3-small',dimensions=1536)
    if index_name in pc.list_indexes().names():
        print(f'Index {index_name} already exists. Loading embeddings..',end='')
        vector_store=Pinecone.from_existing_index(index_name,embeddings)
        print('OK')
    else:
        print(f'Creating index {index_name} and embeddings--',end="")
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws',region='us-east-1')
        
        )

    vector_store=Pinecone.from_documents(chunks,embeddings,index_name="nyp-chatbot")
    
    return vector_store

def save_documents_to_file(all_docs, filename="documents_content.txt"):
    """Save raw content from all documents to a file"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            for i, doc in enumerate(all_docs):
                f.write(f"\n{'='*50}\n")
                f.write(f"DOCUMENT {i+1}\n")
                f.write(f"{'='*50}\n\n")
                
                # Get the content depending on document type
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                else:
                    content = str(doc)
                
                # Write the actual content
                f.write(content)
                f.write("\n\n")
                
        print(f"Documents content saved to {filename}")
        
    except Exception as e:
        print(f"Error saving documents to file: {str(e)}")


async def store_query_embedding(query: str, answer: str, embeddings: OpenAIEmbeddings):
    """Store query-answer pair embeddings in Pinecone"""
    try:
        # Generate a unique ID for this QA pair
        qa_id = str(uuid.uuid4())  # Generate unique identifier
        
        # Create metadata with the unique ID
        metadata = {
            "type": "user_query",
            "qa_id": qa_id,  # Add the unique ID to metadata
            "query": query,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create document with metadata
        qa_doc = Document(
            page_content=f"Question: {query}\nAnswer: {answer}",
            metadata=metadata
        )
        
        # Add to Pinecone
        vector_store.add_documents([qa_doc])
        
        print(f"Stored query embedding with ID {qa_id[:8]}...")  # Print first 8 chars of ID
        
    except Exception as e:
        print(f"Error storing query embedding: {e}")


@app.on_event("startup")
async def startup_event():
    """Initialize the vector store on startup"""
    global vector_store, retriever
    try:
        print("Initializing vector store...")
        all_docs = create_vector_store()
        # save_documents_to_file(all_docs)
        chunks = chunk_data(all_docs, chunk_size=1024)
        # Initialize Pinecone embeddings and vector store
        vector_store = create_embeddings_pinecone(chunks)

        # Initialize retriever
        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 10})
        
        print("Vector store initialized successfully!")
        cleanup_success = await cleanup_old_queries(days_threshold=5)
        if not cleanup_success:
            print("Warning: Query cleanup failed but startup will continue")
    except Exception as e:
        print(f"Error initializing vector store: {e}")

@app.post("/query")
async def query_documents(request: QueryRequest):
    global retriever,vector_store
    try:
        print("\n=== Incoming Request ===")
        print(f"Timestamp: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Query received: '{request.query}'")
        print("=======================\n")

        if retriever is None:
            raise HTTPException(status_code=500, detail="Vector store retriever not initialized.")

        # Initialize OpenAI LLM
        
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=4000 
        )

        
         # Create embeddings instance
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=1536
        )
        similar_queries = vector_store.similarity_search_with_score(
            request.query,
            k=3,
            filter={"type": "user_query"}  # Only look for previous user queries
        )

        # If we found similar queries with good scores (low distance)
        if similar_queries and similar_queries[0][1] < 0.1:  # Adjust threshold as needed
            print("Found similar previous query")
            best_match = similar_queries[0][0]
            return {
                "answer": best_match.metadata.get("answer"),
                "sources": ["Retrieved from previous similar query"]
            }

        

        # Initialize Memory and Prompt
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
        system_template = r'''
            You are a helpful course advisor for DSEB. Use the following pieces of context to answer the user's questions.
            
            CRITICAL RULES:
            1. NEVER start any response with "I don't have..." when information is available
            2. When answering a question, if multiple relevant examples exist in the context, provide them all as a bulleted list.
            3. ALWAYS convert dash-separated or comma-separated lists into bullet points
                For example:
                    Instead of: "The courses are - Engineering - Science - Math"
                    Format as:
                    • Engineering
                    • Science
                    • Math
            4. Remove any disclaimers when you have the information
            5. When answering questions about specific years, pay close attention to keywords like 'Year 1','Year 2', or 'Year 3' to provide the correct information. Do not mix content from different years.
            6. When you are asked about the entry requirement, ask user to provide if they are O-Level, A-Level, N-level or other level student, then provide correct response.
            7. When you don't find specific information, then you MUST:
                • Acknowledge that you don't have the specific information
                • Recommend the user to visit the relevant URL from the list below
                    - For all other diploma course: https://www.nyp.edu.sg/student/study/schools/engineering
                    - For course-related queries: https://www.nyp.edu.sg/student/study/schools/engineering/diploma-sustainability-engineering-business
                    - For certification queries: https://www.nyp.edu.sg/student/study/schools/engineering/stories/engineering-students-to-graduate-with-additional-industry-recognised-certifications
                    - For scholarship information: https://www.nyp.edu.sg/student/study/scholarships-financial-matters/scholarships-study-awards
                    - For course fees: https://www.nyp.edu.sg/student/study/scholarships-financial-matters/fees/annual-course-fees
                    - For financial schemes: https://www.nyp.edu.sg/student/study/scholarships-financial-matters/schemes
                    - For financial aid and bursaries: https://www.nyp.edu.sg/student/study/scholarships-financial-matters/financial-assistance/bursaries
                    - For admission queries: https://www.nyp.edu.sg/student/study/admissions
            If being questioned related to other courses, please mention that you are course advisor for DSEB only and refer to the NYP website for more infromation.
            If still unable to help to provide the information about DSEB, suggest contacting course Manager Dr. Ang Wei Sin for more information.
            ---------------------
            Context: {context}
            '''
        user_template='''
        Question: {question}
        Chat History: {chat_history}
        '''
        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(user_template)
        ]
        qa_prompt = ChatPromptTemplate.from_messages(messages)

        # Create the conversation chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            memory=memory,
            chain_type='stuff',
            combine_docs_chain_kwargs={'prompt': qa_prompt},
            verbose=True,
            output_key="answer",
            max_tokens_limit=4000,
            
        )

        # Get response with simplified input
        result = await qa_chain.ainvoke({"question": request.query})

        #Store query-answer pair embeddings in Pinecone
        await store_query_embedding(
            query=request.query,
            answer=result['answer'],
            embeddings=embeddings
        )

        print(f"Generated response: {result['answer'][:100]}...")

        return {
            "answer": result['answer'],
            "sources": [doc.page_content for doc in result.get('source_documents', [])]
        }

    except Exception as e:
        print("\n=== Error in Request ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("=======================\n")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

async def cleanup_old_queries(days_threshold: int = 30):
    """Remove query embeddings older than threshold days"""
    try:
        print(f"\nAttempting to clean up old queries...")
        
        # Calculate threshold date
        threshold_date = (datetime.now() - timedelta(days=days_threshold)).isoformat()
        
        try:
            # First, try to get all vectors with metadata
            response = vector_store.similarity_search_with_score(
                query="",  # Empty query
                k=100,     # Adjust based on your needs
                filter={
                    "type": "user_query"
                }
            )
            
            if response:
                # Filter old queries locally
                ids_to_delete = []
                for doc, score in response:
                    timestamp = doc.metadata.get('timestamp')
                    if timestamp and timestamp < threshold_date:
                        qa_id = doc.metadata.get('qa_id')
                        if qa_id:
                            ids_to_delete.append(qa_id)
                
                if ids_to_delete:
                    # Delete in batches of 10 to avoid overloading
                    batch_size = 10
                    for i in range(0, len(ids_to_delete), batch_size):
                        batch = ids_to_delete[i:i + batch_size]
                        try:
                            vector_store._index.delete(ids=batch)
                            print(f"Deleted batch of {len(batch)} queries")
                        except Exception as batch_error:
                            print(f"Error deleting batch: {str(batch_error)}")
                            continue
                    
                    print(f"Cleanup completed. Deleted {len(ids_to_delete)} old queries")
                else:
                    print("No old queries found to delete")
            else:
                print("No queries found in the index")
                
        except Exception as delete_error:
            print(f"Error during deletion: {str(delete_error)}")
            print(f"Response headers: {getattr(delete_error, 'response_headers', 'No headers')}")
            print(f"Response body: {getattr(delete_error, 'response_body', 'No body')}")
            return False
            
    except Exception as e:
        print(f"Error in cleanup process: {str(e)}")
        return False
    
    return True

# Serve React static files
# app.mount("/", StaticFiles(directory="build", html=True))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
