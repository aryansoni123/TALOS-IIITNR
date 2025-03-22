# Document Analysis and Chat API

A powerful FastAPI-based application for document analysis, processing, and intelligent chat interactions. This system can process various types of documents including PDFs, audio files, and web content, creating a searchable knowledge base that can be queried through a chat interface.

## Features

- **Document Processing**
  - PDF processing with text and image extraction
  - OCR capabilities for images within PDFs
  - Audio file transcription and analysis
  - Web content processing and link extraction
  - CSV file processing

- **Advanced Search and Retrieval**
  - Vector-based similarity search using FAISS
  - Context-aware responses using LangChain
  - Intelligent document chunking for better context preservation

- **Real-time Processing**
  - File monitoring for automatic updates
  - WebSocket support for progress tracking
  - Concurrent processing of nested web links

## Prerequisites

- Python 3.8+
- Tesseract OCR
- Poppler Utils
- FFmpeg (for audio processing)

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Install system dependencies:

For Windows:
```bash
# Install Tesseract OCR
winget install UB-Mannheim.TesseractOCR

# Install Poppler
pip install pdf2image poppler-utils
```

For Linux:
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils ffmpeg
```

4. Set up your Google API key:
```bash
export GOOGLE_API_KEY=your_api_key_here
```

## Configuration

The application uses several environment variables that can be configured:

- `GOOGLE_API_KEY`: Your Google API key for the Gemini model
- `UPLOAD_DIR`: Directory for uploaded files (default: "uploads")
- `SYSTEM_DIR`: Directory for system files (default: "system")

## Usage

1. Start the server:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

2. Access the API documentation:
```
http://localhost:8000/docs
```

## API Endpoints

- `/upload/pdf`: Upload and process PDF files
- `/upload/audio`: Upload and process audio files
- `/process/link`: Process web links and their content
- `/chat`: Query the knowledge base using natural language

## Example Usage

### Upload a PDF
```python
import requests

files = {'file': open('document.pdf', 'rb')}
response = requests.post('http://localhost:8000/upload/pdf', files=files)
print(response.json())
```

### Chat Query
```python
import requests

query = {'text': 'What are the main points in the document?'}
response = requests.post('http://localhost:8000/chat', json=query)
print(response.json())
```

## Architecture

The system uses several key components:

- **FastAPI**: Web framework for building APIs
- **LangChain**: Framework for developing applications powered by language models
- **FAISS**: Efficient similarity search and clustering of dense vectors
- **Whisper**: Speech recognition model for audio processing
- **Wav2Vec2**: Audio feature extraction
- **PyTesseract**: OCR engine for image text extraction
- **PyMuPDF**: PDF processing library

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI's Whisper for audio transcription
- Facebook's Wav2Vec2 for audio processing
- Google's Gemini model for chat responses
- FAISS by Facebook Research for vector similarity search

## Support

For support, please open an issue in the GitHub repository or contact [your-contact-info].
