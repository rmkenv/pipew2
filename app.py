"""
PipeWrench AI - Web Application
Municipal DPW Knowledge Capture with Document-First Retrieval
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import os
import json
from datetime import datetime
from pathlib import Path
import logging
from anthropic import Anthropic
import requests
from urllib.parse import urlparse
import re
from werkzeug.utils import secure_filename
import secrets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
CORS(app)

# Ensure upload folder exists
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# Configuration
WHITELIST_URL = "https://raw.githubusercontent.com/rmkenv/pipewrench_mvp/main/custom_whitelist.json"
URL_REGEX = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

EMBEDDED_WHITELIST = [
    {"url": "https://www.epa.gov", "description": "EPA Regulations"},
    {"url": "https://www.osha.gov", "description": "OSHA Standards"},
    {"url": "https://www.fhwa.dot.gov", "description": "FHWA Standards"},
    {"url": "https://www.awwa.org", "description": "Water Standards"},
    {"url": "https://www.apwa.net", "description": "APWA Resources"},
    {"url": "https://www.asce.org", "description": "ASCE Standards"},
]

DEPARTMENTS = {
    "general": "General Public Works",
    "water_sewer": "Water & Sewer",
    "streets": "Streets & Highways",
    "environmental": "Environmental Compliance",
    "safety": "Safety & Training",
}

whitelist_urls = []

# PDF extraction
PDF_EXTRACTION_AVAILABLE = False
try:
    import pypdf
    PDF_EXTRACTION_AVAILABLE = True
except:
    try:
        from PyPDF2 import PdfReader
        PDF_EXTRACTION_AVAILABLE = True
    except:
        pass

# Initialize Anthropic client
anthropic_client = None
api_key = os.getenv("ANTHROPIC_API_KEY")
if api_key:
    anthropic_client = Anthropic(api_key=api_key)
    logger.info("✅ Anthropic client initialized")
else:
    logger.warning("⚠️  ANTHROPIC_API_KEY not found")

# Whitelist management
def fetch_whitelist():
    global whitelist_urls
    try:
        response = requests.get(WHITELIST_URL, timeout=10)
        data = response.json()
        whitelist_urls = [entry["url"] for entry in data if "url" in entry]
        logger.info(f"✅ {len(whitelist_urls)} whitelisted URLs loaded")
        return True
    except Exception as e:
        whitelist_urls = [entry["url"] for entry in EMBEDDED_WHITELIST]
        logger.info(f"✅ {len(whitelist_urls)} embedded URLs loaded (fallback)")
        return False

def is_url_whitelisted(url: str) -> bool:
    try:
        parsed = urlparse(url)
        for wl_url in whitelist_urls:
            wl_parsed = urlparse(wl_url)
            if parsed.netloc == wl_parsed.netloc:
                return True
    except:
        pass
    return False

# PDF extraction
def extract_text_from_pdf(file_path: str) -> str:
    if not PDF_EXTRACTION_AVAILABLE:
        return "[PDF library not available]"
    
    try:
        import io
        
        if not Path(file_path).exists():
            return "[File not found]"
        
        with open(file_path, 'rb') as f:
            content = f.read()
        
        text = ""
        
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(content))
            
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                except:
                    pass
        except:
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(io.BytesIO(content))
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    except:
                        pass
            except:
                return "[PDF read error]"
        
        if text.strip():
            return text
        else:
            return "[PDF loaded but no text extracted - may be image-based]"
    
    except Exception as e:
        return f"[Error: {e}]"

# Build system prompt
def build_system_prompt_rag(documents, department, query):
    system_base = """You are a specialized DPW (Department of Public Works) assistant.

CRITICAL: Your information retrieval priority is:
1. PRIMARY: The uploaded document(s) below
2. SECONDARY: Whitelisted reference sources 
3. TERTIARY: General knowledge (only if above don't cover it)

Always cite your sources and explain where you found information."""
    
    prompt = system_base
    prompt += f"\n\nDepartment Context: {DEPARTMENTS.get(department, 'General')}"
    
    if documents:
        prompt += f"\n\n{'='*70}"
        prompt += f"\n🔴 PRIMARY SOURCE: UPLOADED DOCUMENT(S) (CHECK THIS FIRST!)"
        prompt += f"\n{'='*70}\n"
        
        for doc in documents:
            prompt += f"\n<document>\n"
            prompt += f"<name>{doc['name']}</name>\n"
            prompt += f"<content>\n{doc['text'][:8000]}\n</content>\n"
            prompt += f"</document>\n"
        
        prompt += f"\n{'='*70}"
        prompt += f"\n⚠️  IMPORTANT: Search documents FIRST, cite them, then supplement if needed"
        prompt += f"\n{'='*70}\n"
    
    prompt += f"\n<secondary_sources>\n"
    prompt += f"APPROVED REFERENCE SOURCES (use only if document insufficient):\n"
    for i, url in enumerate(whitelist_urls[:5], 1):
        prompt += f"  {i}. {url}\n"
    prompt += f"</secondary_sources>\n"
    
    prompt += """
<response_format>
1. SCRATCHPAD: Internal reasoning about document relevance
2. ANSWER: Clear, well-structured response
3. SOURCES CITED: Document and external sources
4. CONFIDENCE: Assessment based on sources
</response_format>

CRITICAL RULE: Always prioritize document information."""
    
    return prompt

# Initialize whitelist
fetch_whitelist()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/ask', methods=['POST'])
def ask():
    if not anthropic_client:
        return jsonify({'error': 'API key not configured'}), 500
    
    data = request.json
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'Empty question'}), 400
    
    try:
        # Get session data
        documents = session.get('documents', [])
        department = session.get('department', 'general')
        
        # Build system prompt
        system_prompt = build_system_prompt_rag(documents, department, question)
        
        # User message
        user_message = f"""Question: {question}

Remember: Check uploaded documents FIRST, cite them, then supplement if needed."""
        
        # Call Claude
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2000,
            timeout=60.0,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )
        
        answer = response.content[0].text
        
        # Extract cited URLs
        urls = URL_REGEX.findall(answer)
        cleaned_urls = [u.rstrip('.,);]') for u in urls]
        cited = [u for u in cleaned_urls if is_url_whitelisted(u)][:5]
        
        # Store in session history
        if 'history' not in session:
            session['history'] = []
        
        session['history'].append({
            'role': 'user',
            'content': question,
            'timestamp': datetime.now().isoformat()
        })
        session['history'].append({
            'role': 'assistant',
            'content': answer,
            'sources': cited,
            'timestamp': datetime.now().isoformat()
        })
        
        session.modified = True
        
        return jsonify({
            'answer': answer,
            'sources': cited,
            'documents_used': len(documents)
        })
    
    except Exception as e:
        logger.error(f"Error in ask: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files allowed'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text
        text = extract_text_from_pdf(filepath)
        
        # Store in session
        if 'documents' not in session:
            session['documents'] = []
        
        session['documents'].append({
            'name': filename,
            'text': text,
            'loaded_at': datetime.now().isoformat()
        })
        
        session.modified = True
        
        return jsonify({
            'success': True,
            'filename': filename,
            'characters': len(text),
            'total_documents': len(session['documents'])
        })
    
    except Exception as e:
        logger.error(f"Error in upload: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/department', methods=['POST'])
def set_department():
    data = request.json
    dept = data.get('department', 'general')
    
    if dept not in DEPARTMENTS:
        return jsonify({'error': 'Invalid department'}), 400
    
    session['department'] = dept
    session.modified = True
    
    return jsonify({
        'success': True,
        'department': dept,
        'name': DEPARTMENTS[dept]
    })

@app.route('/api/info', methods=['GET'])
def get_info():
    return jsonify({
        'department': session.get('department', 'general'),
        'department_name': DEPARTMENTS.get(session.get('department', 'general')),
        'documents': len(session.get('documents', [])),
        'messages': len(session.get('history', [])),
        'whitelist_urls': len(whitelist_urls),
        'pdf_support': PDF_EXTRACTION_AVAILABLE
    })

@app.route('/api/clear', methods=['POST'])
def clear_session():
    session.clear()
    return jsonify({'success': True})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'api_configured': anthropic_client is not None,
        'pdf_support': PDF_EXTRACTION_AVAILABLE
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
