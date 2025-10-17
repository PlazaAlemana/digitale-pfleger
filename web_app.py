from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from pdf_reader_v2 import PDFMistralAgent
import os

app = Flask(__name__)
CORS(app)

# Initialize the agent once when the server starts
agent = None


def initialize_agent():
    """Initialize the PDF agent on first request."""
    global agent
    if agent is None:
        print("🚀 Initializing PDF Agent...")
        agent = PDFMistralAgent(knowledge_folder="knowledge")

        # Auto-load first PDF
        if agent.available_pdfs:
            agent.load_first_available_pdf()
            print(f"✅ Loaded: {agent.current_pdf}")
        else:
            print("⚠️ No PDFs found in knowledge folder")
    return agent


@app.route('/')
def home():
    """Serve the main HTML page."""
    return render_template('index.html')


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current PDF status and metadata."""
    try:
        agent = initialize_agent()

        if agent.current_pdf:
            return jsonify({
                'success': True,
                'pdf_loaded': True,
                'pdf_name': agent.current_pdf,
                'metadata': agent.pdf_metadata,
                'available_pdfs': agent.available_pdfs
            })
        else:
            return jsonify({
                'success': True,
                'pdf_loaded': False,
                'message': 'No PDF loaded',
                'available_pdfs': agent.available_pdfs
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Handle question and return answer."""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()

        if not question:
            return jsonify({
                'success': False,
                'error': 'Please provide a question'
            }), 400

        agent = initialize_agent()

        if not agent.current_pdf:
            return jsonify({
                'success': False,
                'error': 'No PDF loaded. Please add a PDF to the knowledge folder.'
            }), 400

        print(f"❓ Question: {question}")

        # Get answer from agent
        answer = agent.ask(question)

        return jsonify({
            'success': True,
            'question': question,
            'answer': answer,
            'pdf_name': agent.current_pdf
        })

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/summary', methods=['GET'])
def get_summary():
    """Get PDF summary."""
    try:
        agent = initialize_agent()

        if not agent.current_pdf:
            return jsonify({
                'success': False,
                'error': 'No PDF loaded'
            }), 400

        summary = agent.get_summary()

        return jsonify({
            'success': True,
            'summary': summary,
            'pdf_name': agent.current_pdf
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/load_pdf', methods=['POST'])
def load_pdf():
    """Load a specific PDF by name."""
    try:
        data = request.get_json()
        pdf_name = data.get('pdf_name', '').strip()

        if not pdf_name:
            return jsonify({
                'success': False,
                'error': 'Please provide a PDF name'
            }), 400

        agent = initialize_agent()
        success = agent.load_pdf_by_name(pdf_name)

        if success:
            return jsonify({
                'success': True,
                'message': f'Successfully loaded {pdf_name}',
                'pdf_name': agent.current_pdf,
                'metadata': agent.pdf_metadata
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to load {pdf_name}'
            }), 400

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("=" * 75)
    print("🌐 Digitale Pfleger Web App - Starting...")
    print("=" * 75)

    port = int(os.environ.get('PORT', 5000))

    if os.environ.get('RENDER'):
        # Production mode - use waitress
        print(f"\n📍 Running in PRODUCTION mode on port {port}")
        from waitress import serve

        serve(app, host='0.0.0.0', port=port)
    else:
        # Development mode
        print(f"\n📍 Access the app at: http://localhost:{port}")
        print("📍 API endpoint: http://localhost:{port}/api/ask")
        print("\n💡 Press Ctrl+C to stop the server\n")
        app.run(debug=True, host='0.0.0.0', port=port)