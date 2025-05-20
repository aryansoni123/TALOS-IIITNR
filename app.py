from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import whisper
import librosa
import numpy as np
import torch
import faiss
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import tempfile
from typing import List, Optional, Dict, Set, Tuple
from datetime import datetime
from pdf2image import convert_from_path
import pytesseract
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from PIL import Image
import io
import fitz  # PyMuPDF
import concurrent.futures
import threading
import zipfile
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

# Initialize FastAPI app
app = FastAPI(title="Document Analysis API")

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Configuration
API_KEY = "AIzaSyBz1c99bSI4EBEh9-Th_JOLm2xM_y8OzUw"
UPLOAD_DIR = "uploads"  # Directory to store uploaded files
SYSTEM_DIR = "system"

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SYSTEM_DIR, exist_ok=True)

# Initialize models and processors
whisper_model = whisper.load_model("base")
wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize FAISS indices
db = None
audio_text_db = None
audio_index = faiss.IndexFlatL2(768)

# Pydantic models for request/response
class Query(BaseModel):
    text: str

class RetrievalInfo(BaseModel):
    source_type: str
    content: str
    metadata: Dict
    similarity_score: float
    timestamp: str

class ChatResponse(BaseModel):
    response: str
    retrieval_details: List[RetrievalInfo]
    reasoning: str
    confidence_score: float

class WebLink(BaseModel):
    url: str

class FileUpdateHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            print(f"File {event.src_path} has been modified")
            update_vector_store(event.src_path)

class LinkTracker:
    def __init__(self):
        self.processed_links = set()
        self.link_content = {}
        self.lock = threading.Lock()

    def add_link(self, url: str, content: str):
        with self.lock:
            self.processed_links.add(url)
            self.link_content[url] = content

    def is_processed(self, url: str) -> bool:
        with self.lock:
            return url in self.processed_links

    def get_content(self, url: str) -> Optional[str]:
        with self.lock:
            return self.link_content.get(url)

# Initialize link tracker
link_tracker = LinkTracker()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

class BulkUploadProgress:
    def __init__(self):
        self.total_files = 0
        self.processed_files = 0
        self.failed_files = []
        self.current_file = ""
        self.start_time = None

    def to_dict(self):
        return {
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "current_file": self.current_file,
            "progress_percentage": (self.processed_files / self.total_files * 100) if self.total_files > 0 else 0
        }

async def process_file_bulk(file_path: str, progress: BulkUploadProgress, manager: ConnectionManager):
    """Process a single file in bulk upload."""
    try:
        progress.current_file = os.path.basename(file_path)
        await manager.broadcast(json.dumps(progress.to_dict()))

        if file_path.endswith('.pdf'):
            docs = load_pdf(file_path)
        elif file_path.endswith('.csv'):
            docs = load_csv(file_path)
        elif file_path.endswith(('.mp3', '.wav')):
            docs = process_audio_file(file_path)
        else:
            progress.failed_files.append(f"{progress.current_file} - Unsupported file type")
            return None

        progress.processed_files += 1
        await manager.broadcast(json.dumps(progress.to_dict()))
        return docs
    except Exception as e:
        progress.failed_files.append(f"{progress.current_file} - {str(e)}")
        await manager.broadcast(json.dumps(progress.to_dict()))
        return None

@app.websocket("/ws/bulk-upload")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            # Echo back to confirm connection is alive
            await websocket.send_text(json.dumps({"status": "connected"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)

@app.post("/upload/bulk")
async def bulk_upload(file: UploadFile = File(...)):
    """Handle bulk upload of ZIP file containing multiple documents."""
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only ZIP files are allowed")

    progress = BulkUploadProgress()
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "upload.zip")

    try:
        # Save ZIP file
        with open(zip_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Extract ZIP
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Count total files
            progress.total_files = len([f for f in zip_ref.namelist() if not f.endswith('/')])
            progress.start_time = datetime.now()
            
            # Broadcast initial progress
            await manager.broadcast(json.dumps(progress.to_dict()))

            # Extract files
            zip_ref.extractall(temp_dir)

        # Process files in batches
        batch_size = 100
        all_docs = []
        
        for root, _, files in os.walk(temp_dir):
            valid_files = [os.path.join(root, f) for f in files 
                         if f.endswith(('.pdf', '.csv', '.mp3', '.wav')) 
                         and not f == "upload.zip"]
            
            # Process files in batches
            for i in range(0, len(valid_files), batch_size):
                batch = valid_files[i:i + batch_size]
                tasks = [process_file_bulk(f, progress, manager) for f in batch]
                batch_results = await asyncio.gather(*tasks)
                
                # Add successful results to vector store
                batch_docs = [doc for docs in batch_results if docs for doc in docs]
                if batch_docs:
                    all_docs.extend(batch_docs)

                # Broadcast progress after each batch
                await manager.broadcast(json.dumps(progress.to_dict()))

        # Update vector store with all documents
        if all_docs:
            global db
            if db is None:
                db = FAISS.from_documents(all_docs, embeddings)
            else:
                new_db = FAISS.from_documents(all_docs, embeddings)
                db.merge_from(new_db)

        # Final progress update
        await manager.broadcast(json.dumps(progress.to_dict()))

        return {
            "message": "Bulk upload completed",
            "total_files": progress.total_files,
            "processed_files": progress.processed_files,
            "failed_files": progress.failed_files,
            "processing_time": str(datetime.now() - progress.start_time)
        }

    except Exception as e:
        print(f"Bulk upload error: {str(e)}")
        # Send error status through WebSocket
        await manager.broadcast(json.dumps({
            "error": str(e),
            "status": "failed"
        }))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary directory
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Error cleaning up temporary files: {str(e)}")

def update_vector_store(file_path):
    """Update vector store when a file is modified."""
    global db
    try:
        # Determine file type and process accordingly
        if file_path.endswith('.pdf'):
            docs = load_pdf(file_path)
        elif file_path.endswith('.csv'):
            docs = load_csv(file_path)
        elif file_path.endswith(('.mp3', '.wav')):
            docs = process_audio_file(file_path)
        else:
            return

        # Update vector store
        if db is not None:
            new_db = FAISS.from_documents(docs, embeddings)
            db.merge_from(new_db)
            print(f"Vector store updated for {file_path}")

    except Exception as e:
        print(f"Error updating vector store: {str(e)}")

def initialize_system():
    """Initialize system with a default document to prevent empty database errors."""
    global db
    
    system_file = os.path.join(SYSTEM_DIR, "system_info.txt")
    
    # Create system info document if it doesn't exist
    if not os.path.exists(system_file):
        system_content = """
        Welcome to the AI Assistant System
        
        This system can help you with:
        1. Processing and analyzing documents (PDF, CSV, audio files)
        2. Answering questions about uploaded content
        3. Processing web links and their content
        4. Extracting information from images in PDFs
        
        The system maintains a vector database of all processed content for efficient retrieval.
        """
        
        with open(system_file, "w") as f:
            f.write(system_content)
    
    # Load system content into vector store
    try:
        with open(system_file, "r") as f:
            content = f.read()
        
        doc = Document(
            page_content=content,
            metadata={"source": "system", "type": "system_info"}
        )
        
        if db is None:
            db = FAISS.from_documents([doc], embeddings)
        else:
            new_db = FAISS.from_documents([doc], embeddings)
            db.merge_from(new_db)
        
        print("System initialized with default content")
        
        # Set up file monitoring
        event_handler = FileUpdateHandler()
        observer = Observer()
        observer.schedule(event_handler, UPLOAD_DIR, recursive=False)
        observer.start()
        print("File monitoring started")
        
    except Exception as e:
        print(f"Error initializing system: {str(e)}")

# Initialize system on startup
initialize_system()

# Helper functions
def extract_text_and_images_from_pdf(pdf_path: str) -> List[Document]:
    """Extract both text and images from PDF using PyMuPDF."""
    try:
        docs = []
        pdf_document = fitz.open(pdf_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Extract text
            text = page.get_text()
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(pdf_path),
                        "page": page_num + 1,
                        "type": "text"
                    }
                ))
            
            # Extract images
            image_list = page.get_images(full=True)
            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert image to PIL Image for OCR
                image = Image.open(io.BytesIO(image_bytes))
                
                # Perform OCR on the image
                try:
                    text = pytesseract.image_to_string(image)
                    if text.strip():
                        docs.append(Document(
                            page_content=text,
                            metadata={
                                "source": os.path.basename(pdf_path),
                                "page": page_num + 1,
                                "image_index": img_index,
                                "type": "image_text"
                            }
                        ))
                except Exception as e:
                    print(f"OCR error for image {img_index} on page {page_num + 1}: {str(e)}")
            
            # Extract links
            links = page.get_links()
            for link in links:
                if "uri" in link:
                    url = link["uri"]
                    if url.startswith(("http://", "https://")):
                        try:
                            link_docs = process_web_link(url, depth=1)
                            docs.extend(link_docs)
                        except Exception as e:
                            print(f"Error processing link {url}: {str(e)}")
        
        return docs
    except Exception as e:
        print(f"Error in PDF processing: {str(e)}")
        return []

def process_web_link(url: str, depth: int = 0, max_depth: int = 2) -> List[Document]:
    """Process web links recursively up to max_depth."""
    if depth > max_depth or link_tracker.is_processed(url):
        return []
    
    try:
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL format")
        
        # Fetch webpage content
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text(separator='\n')
        
        # Add to link tracker
        link_tracker.add_link(url, text)
        
        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.create_documents([text])
        
        # Add metadata
        for doc in docs:
            doc.metadata.update({
                "source": url,
                "type": "web_content",
                "title": soup.title.string if soup.title else url,
                "depth": depth
            })
        
        # Process nested links if not at max depth
        if depth < max_depth:
            nested_docs = []
            links = soup.find_all('a', href=True)
            valid_links = [
                link['href'] 
                for link in links 
                if link['href'].startswith(('http://', 'https://'))
            ]
            
            # Process nested links in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                future_to_url = {
                    executor.submit(process_web_link, link, depth + 1): link 
                    for link in valid_links[:5]  # Limit to 5 nested links per page
                }
                for future in concurrent.futures.as_completed(future_to_url):
                    try:
                        nested_docs.extend(future.result())
                    except Exception as e:
                        print(f"Error processing nested link: {str(e)}")
            
            docs.extend(nested_docs)
        
        return docs
    except Exception as e:
        print(f"Error processing web link {url}: {str(e)}")
        return []

def load_pdf(file_path: str) -> List[Document]:
    """Load and process PDF document with enhanced image and link handling."""
    try:
        # Extract text, images, and links
        docs = extract_text_and_images_from_pdf(file_path)
        
        # Add filename to metadata
        filename = os.path.basename(file_path)
        for doc in docs:
            doc.metadata["source"] = filename
        
        return docs
    except Exception as e:
        print(f"Error loading PDF: {str(e)}")
        return []

def load_csv(file_path: str) -> List[Document]:
    """Load CSV file and convert to document chunks."""
    df = pd.read_csv(file_path)
    text_data = "\n".join(df.apply(lambda row: " | ".join(map(str, row)), axis=1))
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text_data])
    # Add filename to metadata
    filename = os.path.basename(file_path)
    for doc in docs:
        doc.metadata["source"] = filename
    return docs

def load_audio(file_path: str, embeddings) -> tuple:
    """Process audio file to extract text, timestamps, and embeddings."""
    result = whisper_model.transcribe(file_path, word_timestamps=True)
    
    audio, sr = librosa.load(file_path, sr=16000)
    
    # Generate audio embeddings using Wav2Vec2
    inputs = wav2vec_processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        audio_embedding = wav2vec_model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
    
    segments = result.get("segments", [])
    audio_docs = []
    audio_embeddings = []
    
    # Get filename
    filename = os.path.basename(file_path)
    
    for seg in segments:
        start, end = seg['start'], seg['end']
        text = seg['text']
        
        text_embedding = embeddings.embed_query(text)
        
        audio_doc = {
            "page_content": f"[{start:.2f}s - {end:.2f}s] {text}",
            "metadata": {
                "source": filename,
                "start": start,
                "end": end
            }
        }
        
        audio_docs.append(audio_doc)
        audio_embeddings.append(text_embedding)
    
    return audio_docs, np.array(audio_embeddings), audio_embedding

# API Endpoints
@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Handle PDF file upload and processing."""
    try:
        # Save uploaded file temporarily
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process PDF
        pdf_docs = load_pdf(file_path)
        global db
        if db is None:
            db = FAISS.from_documents(pdf_docs, embeddings)
        else:
            new_db = FAISS.from_documents(pdf_docs, embeddings)
            db.merge_from(new_db)
        
        return {"message": "PDF processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/audio")
async def upload_audio(file: UploadFile = File(...)):
    """Handle audio file upload and processing."""
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process audio
        audio_docs, audio_text_embeddings, audio_embedding = load_audio(file_path, embeddings)
        global audio_text_db, audio_index
        
        if audio_text_db is None:
            audio_text_db = FAISS.from_embeddings(
                list(zip([doc["page_content"] for doc in audio_docs], audio_text_embeddings)), 
                embeddings
            )
        else:
            new_audio_text_db = FAISS.from_embeddings(
                list(zip([doc["page_content"] for doc in audio_docs], audio_text_embeddings)), 
                embeddings
            )
            audio_text_db.merge_from(new_audio_text_db)
        
        audio_index.add(np.array([audio_embedding]))
        
        return {"message": "Audio processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process/link")
async def process_link(link: WebLink):
    """Handle web link processing."""
    try:
        docs = process_web_link(link.url)
        if not docs:
            raise HTTPException(status_code=400, detail="No content could be extracted from the URL")
        
        global db
        if db is None:
            db = FAISS.from_documents(docs, embeddings)
        else:
            new_db = FAISS.from_documents(docs, embeddings)
            db.merge_from(new_db)
        
        return {"message": "Web link processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(query: Query):
    """Handle chat queries and return responses with detailed retrieval information."""
    
    # Handle greetings
    greetings = ["hi", "hello", "hey", "greetings"]
    if query.text.lower().strip() in greetings:
        return ChatResponse(
            response="Hello! How can I help you today?",
            retrieval_details=[],
            reasoning="This is a greeting response",
            confidence_score=1.0
        )
    
    if db is None:
        raise HTTPException(
            status_code=400,
            detail="Please upload documents or provide web links first"
        )
    
    retrieval_details = []
    
    try:
        # Use similarity_search_with_score with a threshold
        docs_with_scores = db.similarity_search_with_score(query.text, k=5)
        print(f"Retrieved {len(docs_with_scores)} documents")
        
        # Filter documents by similarity threshold
        similarity_threshold = 0.7
        filtered_docs = [
            (doc, score) for doc, score in docs_with_scores 
            if score > similarity_threshold
        ]
        
        if not filtered_docs:
            return ChatResponse(
                response="I couldn't find any sufficiently relevant information to answer your question accurately. Could you please rephrase your question or provide more context?",
                retrieval_details=[],
                reasoning="No documents met the relevance threshold.",
                confidence_score=0.0
            )
        
        for doc, score in filtered_docs:
            source_type = doc.metadata.get("type", "document")
            retrieval_details.append(
                RetrievalInfo(
                    source_type=source_type,
                    content=doc.page_content,
                    metadata=doc.metadata,
                    similarity_score=float(score),
                    timestamp=datetime.now().isoformat()
                )
            )
    except Exception as e:
        print(f"Error in similarity search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in similarity search: {str(e)}")
    
    # Sort retrieval details by similarity score (higher scores first)
    retrieval_details.sort(key=lambda x: x.similarity_score, reverse=True)
    
    # Prepare context for LLM
    context_docs = [
        f"Source ({detail.source_type}): {detail.content}\nConfidence: {detail.similarity_score:.2f}"
        for detail in retrieval_details[:3]  # Use top 3 most relevant documents
    ]
    context = "\n\n".join(context_docs)
    
    try:
        system_prompt = """You are a professional AI assistant that provides accurate information based solely on the given context. Follow these rules strictly:

1. Only use information explicitly stated in the provided context
2. If the context doesn't contain enough information to answer fully, acknowledge the limitations
3. If you're unsure about any part of the answer, express that uncertainty
4. Do not make assumptions or add information beyond what's in the context
5. If you can't answer the question from the context, say so directly

Format your response in clear, natural language without using markdown or special formatting."""

        user_prompt = f"Question: {query.text}\n\nAvailable Context:\n{context}\n\nProvide an answer using only the information from the context provided above. If you cannot answer from this context, say so clearly."
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=API_KEY,
            temperature=0.3,  # Lower temperature for more focused responses
            top_p=0.7,        # Reduce randomness in token selection
            top_k=40          # Limit token selection pool
        )
        
        result = llm.invoke(f"{system_prompt}\n\n{user_prompt}")
        
        # Calculate confidence score based on document relevance
        confidence_score = min(1.0, sum([detail.similarity_score for detail in retrieval_details[:3]]) / 3)
        
        # Clean up the response text
        response_text = result.content if hasattr(result, "content") else str(result)
        response_text = response_text.replace("**", "").replace("##", "").replace("*", "")
        
        return ChatResponse(
            response=response_text,
            retrieval_details=retrieval_details[:3],  # Only return top 3 most relevant sources
            reasoning="Response generated from highly relevant documents with strict adherence to context.",
            confidence_score=confidence_score
        )
        
    except Exception as e:
        print(f"Error in LLM response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in generating response: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 