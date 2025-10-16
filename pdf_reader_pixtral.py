import os
import re
import sys
import time
import base64
import io
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

# Validate imports with helpful error messages
try:
    import PyPDF2
except ImportError:
    print("❌ PyPDF2 not installed. Run: pip install PyPDF2")
    sys.exit(1)

try:
    from mistralai import Mistral
except ImportError:
    print("❌ mistralai not installed. Run: pip install mistralai")
    sys.exit(1)

try:
    from pdf2image import convert_from_path
except ImportError:
    print("⚠️  pdf2image not installed. Pixtral extraction will not work.")
    print("   Install with: pip install pdf2image")


class PDFMistralAgent:
    """
    AI Agent that reads PDF documents and answers questions using Mistral AI.
    Handles large PDFs with intelligent chunking and context retrieval.
    Automatically loads PDFs from the 'knowledge' folder.
    """

    def __init__(self, knowledge_folder: str = "knowledge", chunk_size: int = 2500, chunk_overlap: int = 300,
                 extraction_model: str = "pixtral-large"):
        """
        Initialize the PDF Mistral Agent.

        Args:
            knowledge_folder: Folder containing PDF files (default: "knowledge")
            chunk_size: Size of text chunks (default: 2500 chars)
            chunk_overlap: Overlap between chunks (default: 300 chars)
            extraction_model: "pixtral-large" or "pixtral-12b"
        """
        load_dotenv()

        # Validate API key
        self.api_key = os.getenv('MISTRAL_API_KEY')
        if not self.api_key:
            raise ValueError(
                "❌ MISTRAL_API_KEY not found!\n"
                "Please create a .env file with: MISTRAL_API_KEY=your_key_here"
            )

        # Initialize Mistral client with error handling
        try:
            self.client = Mistral(api_key=self.api_key)
            self.model = "mistral-small-latest"
        except Exception as e:
            raise ValueError(f"❌ Failed to initialize Mistral client: {str(e)}")

        # Configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extraction_model = extraction_model
        self.vision_model = "pixtral-large-latest" if extraction_model == "pixtral-large" else "pixtral-12b-latest"

        # Set knowledge folder path (relative to project root)
        self.knowledge_folder = self._get_knowledge_folder_path(knowledge_folder)

        # Try to find poppler path on Windows
        self.poppler_path = self._find_poppler_path()

        # Storage
        self.pdf_content: str = ""
        self.chunks: List[Tuple[str, int]] = []  # (chunk_text, start_char)
        self.pdf_metadata: Dict = {}
        self.current_pdf: Optional[str] = None
        self.available_pdfs: List[str] = []

        print("✅ PDF Mistral Agent initialized successfully!")
        print(f"   Model: {self.model}")
        print(f"   Chunk size: {self.chunk_size} chars")
        print(f"   Knowledge folder: {self.knowledge_folder}")
        print(f"   Extraction model: {self.extraction_model}")
        if self.poppler_path:
            print(f"   Poppler path: {self.poppler_path}")

        # Discover available PDFs
        self._discover_pdfs()

    def _find_poppler_path(self) -> Optional[str]:
        """Find poppler installation path on Windows (supports German & English Windows)."""
        # Extended list of possible paths including German Windows paths
        possible_paths = [
            # English Windows paths
            r"C:\Program Files\poppler\Library\bin",
            r"C:\poppler\Library\bin",
            r"C:\Program Files (x86)\poppler\Library\bin",
            # German Windows paths
            r"C:\Programme\poppler\Library\bin",
            r"C:\Programme (x86)\poppler\Library\bin",
            # Version-specific paths (common installation pattern)
            r"C:\Program Files\poppler-25.07.0\Library\bin",
            r"C:\Programme\poppler-25.07.0\Library\bin",
            r"C:\Program Files\poppler-24.08.0\Library\bin",
            r"C:\Programme\poppler-24.08.0\Library\bin",
        ]

        # Also check for any poppler installation in common directories
        for base_dir in [r"C:\Program Files", r"C:\Programme", r"C:\Program Files (x86)", r"C:\Programme (x86)"]:
            if os.path.exists(base_dir):
                try:
                    for item in os.listdir(base_dir):
                        if item.lower().startswith('poppler'):
                            poppler_bin = os.path.join(base_dir, item, "Library", "bin")
                            if os.path.exists(poppler_bin):
                                possible_paths.append(poppler_bin)
                except (PermissionError, OSError):
                    pass

        # Check each possible path
        for path in possible_paths:
            if os.path.exists(path):
                print(f"   ✅ Found Poppler at: {path}")
                return path

        # Check if poppler is in PATH
        try:
            import subprocess
            result = subprocess.run(['pdftoppm', '-v'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("   ✅ Poppler found in system PATH")
                return None  # poppler is in PATH, no need to specify path
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        except Exception as e:
            print(f"   ⚠️  Error checking PATH for Poppler: {str(e)}")

        print("   ⚠️  Poppler not found in common locations or PATH")
        return None

    def _get_knowledge_folder_path(self, folder_name: str) -> str:
        """
        Get the absolute path to the knowledge folder.

        Args:
            folder_name: Name of the knowledge folder

        Returns:
            Absolute path to the knowledge folder
        """
        script_dir = Path(__file__).parent.absolute()
        knowledge_path = script_dir / folder_name

        if not knowledge_path.exists():
            knowledge_path = script_dir.parent / folder_name

        if not knowledge_path.exists():
            print(f"⚠️  Warning: '{folder_name}' folder not found.")
            print(f"   Creating folder at: {script_dir / folder_name}")
            (script_dir / folder_name).mkdir(exist_ok=True)
            knowledge_path = script_dir / folder_name

        return str(knowledge_path)

    def _discover_pdfs(self) -> None:
        """Discover all PDF files in the knowledge folder."""
        try:
            knowledge_path = Path(self.knowledge_folder)

            if not knowledge_path.exists():
                print(f"⚠️  Knowledge folder not found: {self.knowledge_folder}")
                self.available_pdfs = []
                return

            pdf_files = list(knowledge_path.glob("*.pdf"))
            self.available_pdfs = [pdf.name for pdf in pdf_files]

            if self.available_pdfs:
                print(f"\n📚 Found {len(self.available_pdfs)} PDF(s) in knowledge folder:")
                for i, pdf_name in enumerate(self.available_pdfs, 1):
                    print(f"   {i}. {pdf_name}")
            else:
                print(f"\n⚠️  No PDF files found in: {self.knowledge_folder}")
                print("   Please add PDF files to the knowledge folder.")

        except Exception as e:
            print(f"⚠️  Error discovering PDFs: {str(e)}")
            self.available_pdfs = []

    def list_available_pdfs(self) -> List[str]:
        """
        List all available PDFs in the knowledge folder.

        Returns:
            List of PDF filenames
        """
        self._discover_pdfs()
        return self.available_pdfs

    def load_pdf_by_name(self, filename: str) -> bool:
        """
        Load a PDF by filename from the knowledge folder.

        Args:
            filename: Name of the PDF file (with or without .pdf extension)

        Returns:
            bool: True if successful, False otherwise
        """
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'

        pdf_path = os.path.join(self.knowledge_folder, filename)
        return self.load_pdf(pdf_path)

    def load_first_available_pdf(self) -> bool:
        """
        Automatically load the first available PDF from the knowledge folder.

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.available_pdfs:
            print("❌ No PDFs available in knowledge folder.")
            return False

        print(f"📄 Auto-loading first available PDF: {self.available_pdfs[0]}")
        return self.load_pdf_by_name(self.available_pdfs[0])

    def _extract_text_with_pixtral_large(self, pdf_path: str) -> str:
        """
        Extract content from complex multi-column PDFs using Pixtral Large.
        Converts PDF pages to images first, then processes with vision model.
        """
        print("   🔍 Using Pixtral Large (124B) for complex document extraction...")
        print("   ⏳ Converting PDF pages to images...")

        try:
            # Convert PDF to images (one per page)
            print("   🔍 Looking for Poppler...")

            if self.poppler_path:
                print(f"   📍 Using Poppler from: {self.poppler_path}")
                images = convert_from_path(pdf_path, dpi=200, poppler_path=self.poppler_path)
            else:
                print("   🔍 Trying system PATH for Poppler...")
                images = convert_from_path(pdf_path, dpi=200)

            print(f"   📄 Converted {len(images)} pages to images")
            print("   🔍 Processing pages with Pixtral Large...")

            all_extracted_text = []

            # Process each page image
            for page_num, image in enumerate(images, start=1):
                print(f"   📖 Processing page {page_num}/{len(images)}...")

                # Convert PIL Image to base64
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                # Use Pixtral Large via chat completions API
                response = self.client.chat.complete(
                    model=self.vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"""Extract ALL text from this page (Page {page_num}) with maximum accuracy.

CRITICAL REQUIREMENTS:
- Preserve ALL columns (left, middle, AND right columns)
- Maintain table structure and relationships
- Extract headers, footers, and annotations
- Preserve formatting and layout
- Include ALL contact information and addresses
- Do NOT skip any sections

Output the complete text in a structured format that preserves the page's layout."""
                                },
                                {
                                    "type": "image_url",
                                    "image_url": f"data:image/png;base64,{img_base64}"
                                }
                            ]
                        }
                    ],
                    temperature=0.0,
                    max_tokens=4000
                )

                page_text = response.choices[0].message.content
                all_extracted_text.append(f"[PAGE {page_num}]\n{page_text}")

                # Small delay to avoid rate limits
                if page_num < len(images):
                    time.sleep(1)

            extracted_text = "\n\n".join(all_extracted_text)

            if not extracted_text or not extracted_text.strip():
                raise Exception("No text extracted from Pixtral response")

            print(f"   ✅ Pixtral Large extraction completed!")
            print(f"   📊 Extracted {len(extracted_text)} characters from {len(images)} pages")

            return extracted_text.strip()

        except ImportError:
            print("   ❌ pdf2image not installed!")
            print("   💡 Install with: pip install pdf2image")
            print("   💡 Also install Poppler:")
            print("      - Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases")
            print("      - Extract to C:\\Program Files\\poppler\\")
            raise Exception("pdf2image library not installed")
        except Exception as e:
            print(f"   ❌ Pixtral Large Error: {str(e)}")
            if "Unable to get page count" in str(e):
                print("   💡 Poppler not found! Please install:")
                print("      1. Download from: https://github.com/oschwartz10612/poppler-windows/releases/")
                print("      2. Extract to: C:\\Program Files\\poppler\\ or C:\\Programme\\poppler\\")
                print("      3. Add C:\\Program Files\\poppler\\Library\\bin to your system PATH")
                print("      4. Restart your computer or IDE")
            raise Exception(f"Pixtral Large extraction failed: {str(e)}")

    def load_pdf(self, pdf_path: str) -> bool:
        """
        Load PDF with intelligent extraction strategy.
        """
        try:
            # Clean and normalize path
            pdf_path = pdf_path.strip().strip('"').strip("'")

            if not os.path.isabs(pdf_path):
                if not os.path.exists(pdf_path):
                    pdf_path = os.path.join(self.knowledge_folder, pdf_path)

            pdf_path = os.path.abspath(pdf_path)

            # Validate file exists
            if not os.path.exists(pdf_path):
                print(f"❌ Error: File not found")
                print(f"   Looking for: {pdf_path}")
                return False

            # Validate it's a PDF
            if not pdf_path.lower().endswith('.pdf'):
                print("❌ Error: File must be a PDF (.pdf extension)")
                return False

            # Check file size
            file_size = os.path.getsize(pdf_path)
            file_size_mb = file_size / (1024 * 1024)

            print(f"\n📄 Loading PDF: {os.path.basename(pdf_path)}")
            print(f"   Size: {file_size_mb:.2f} MB")

            if file_size_mb > 50:
                print("⚠️  Warning: Large PDF detected. This may take a while...")

            # Strategy 1: Try PyPDF2 first (fastest)
            self.pdf_content = self._extract_pdf_text_enhanced(pdf_path)

            if not self.pdf_content.strip():
                print("   ⚠️  No text with PyPDF2")

                # Strategy 2: Try Pixtral Large for complex documents
                try:
                    print("   🔄 Attempting Pixtral Large (best for complex layouts)...")
                    self.pdf_content = self._extract_text_with_pixtral_large(pdf_path)
                except Exception as pixtral_error:
                    print(f"   ❌ Pixtral failed: {str(pixtral_error)}")
                    print("   💡 This PDF appears to be image-based or scanned.")
                    print("   💡 Required dependencies for Pixtral:")
                    print("      - pip install pdf2image")
                    print("      - Install poppler (see instructions above)")
                    print("   💡 Alternative: Convert PDF to text using online tools")
                    return False

            if not self.pdf_content.strip():
                print("❌ Error: Could not extract any text from PDF")
                print("   This PDF may be image-based, scanned, or protected.")
                return False

            # Create chunks with positions
            self.chunks = self._create_chunks(self.pdf_content)

            # Store metadata
            self.pdf_metadata = {
                'filename': os.path.basename(pdf_path),
                'path': pdf_path,
                'size_mb': round(file_size_mb, 2),
                'num_chunks': len(self.chunks),
                'total_chars': len(self.pdf_content),
                'total_words': len(self.pdf_content.split()),
                'loaded_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            self.current_pdf = os.path.basename(pdf_path)

            print(f"✅ PDF loaded successfully!")
            print(f"   📊 Characters: {self.pdf_metadata['total_chars']:,}")
            print(f"   📝 Words: {self.pdf_metadata['total_words']:,}")
            print(f"   📦 Chunks: {self.pdf_metadata['num_chunks']}")

            return True

        except PyPDF2.errors.PdfReadError:
            print("❌ Error: Corrupted or invalid PDF file")
            return False
        except PermissionError:
            print("❌ Error: Permission denied. File may be open in another program")
            return False
        except Exception as e:
            print(f"❌ Error loading PDF: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _extract_pdf_text_enhanced(self, pdf_path: str) -> str:
        """Enhanced PDF text extraction with multiple methods."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)

                print(f"   📖 Extracting text from {total_pages} pages...")

                all_text = []
                extracted_pages = 0

                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    try:
                        # Method 1: Standard extraction
                        text = page.extract_text() or ""

                        # Method 2: Try alternative extraction if standard fails
                        if not text.strip():
                            text = self._extract_text_alternative(page)

                        if text.strip():
                            text = f"[PAGE {page_num}]\n{text.strip()}"
                            all_text.append(text)
                            extracted_pages += 1
                    except Exception as e:
                        print(f"   ⚠️  Failed to extract page {page_num}: {str(e)}")
                        continue

                print(f"   ✅ Successfully extracted {extracted_pages}/{total_pages} pages")

                if not all_text:
                    return ""

                full_content = "\n\n".join(all_text)
                # Clean up the text
                full_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', full_content)
                full_content = re.sub(r'[ \t]+', ' ', full_content)

                return full_content.strip()

        except Exception as e:
            print(f"   ⚠️  Standard PDF extraction failed: {str(e)}")
            return ""

    def _extract_text_alternative(self, page) -> str:
        """Alternative text extraction method for problematic PDFs."""
        try:
            # Try different extraction methods for problematic PDFs
            text = page.extract_text()
            if text and text.strip():
                return text

            # If standard extraction fails, return empty string
            return ""
        except:
            return ""

    def _create_chunks(self, text: str) -> List[Tuple[str, int]]:
        """
        Split text into overlapping chunks with position tracking.

        Args:
            text: Full text to chunk

        Returns:
            List of (chunk_text, start_position) tuples
        """
        if not text.strip():
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size

            if end < text_length:
                # Try to break at sentence boundary
                for punct in ['. ', '! ', '? ', '\n\n']:
                    sentence_end = text.rfind(punct, start + self.chunk_size // 2, end)
                    if sentence_end > start:
                        end = sentence_end + len(punct)
                        break
                else:
                    # Try word boundary
                    space = text.rfind(' ', start, end)
                    if space > start + self.chunk_size // 2:
                        end = space

            chunk = text[start:end].strip()
            if chunk:
                chunks.append((chunk, start))

            start = end - self.chunk_overlap
            if start >= text_length:
                break

        return chunks

    def _find_relevant_chunks(self, question: str, top_k: int = 4) -> List[str]:
        """
        Find most relevant chunks using improved keyword matching and position.

        Args:
            question: The user's question
            top_k: Number of top chunks to return

        Returns:
            List of relevant chunks
        """
        if not self.chunks:
            return []

        common_words = {'what', 'where', 'when', 'which', 'who', 'how', 'does',
                        'this', 'that', 'these', 'those', 'with', 'from', 'about',
                        'have', 'been', 'were', 'their', 'there', 'would', 'could'}
        keywords = [w.lower() for w in re.findall(r'\b\w{4,}\b', question.lower())
                    if w.lower() not in common_words]

        if not keywords:
            return [chunk for chunk, _ in self.chunks[:top_k]]

        chunk_scores = []
        for i, (chunk, position) in enumerate(self.chunks):
            chunk_lower = chunk.lower()
            keyword_score = sum(chunk_lower.count(kw) for kw in keywords)
            phrase_bonus = 10 if any(kw in chunk_lower for kw in keywords) else 0
            position_bonus = max(0, 5 - (i / len(self.chunks)) * 5)
            total_score = keyword_score + phrase_bonus + position_bonus
            chunk_scores.append((total_score, i, chunk))

        chunk_scores.sort(reverse=True, key=lambda x: x[0])
        top_chunks = [chunk for _, _, chunk in chunk_scores[:top_k]]

        if chunk_scores[0][0] == 0:
            top_chunks = [chunk for chunk, _ in self.chunks[:top_k]]

        return top_chunks

    def ask(self, question: str, use_full_context: bool = False,
            max_retries: int = 2) -> str:
        """
        Ask a question about the loaded PDF with retry logic.

        Args:
            question: Question to ask
            use_full_context: Use entire PDF (slower but comprehensive)
            max_retries: Number of retries on API errors

        Returns:
            Answer string
        """
        if not self.current_pdf:
            return "❌ No PDF loaded. Please load a PDF first."

        if not question.strip():
            return "❌ Please provide a valid question."

        for attempt in range(max_retries + 1):
            try:
                if use_full_context or len(self.pdf_content) < 8000:
                    context = self.pdf_content
                    print("🔍 Using full PDF context...")
                else:
                    relevant_chunks = self._find_relevant_chunks(question, top_k=4)
                    context = "\n\n---\n\n".join(relevant_chunks)
                    print(f"🔍 Using {len(relevant_chunks)} relevant chunks...")

                max_context_length = 25000
                if len(context) > max_context_length:
                    context = context[:max_context_length] + "\n\n[... content truncated ...]"
                    print(f"   ⚠️  Context truncated to {max_context_length} chars")

                system_prompt = """You are an expert PDF document analyst. Answer questions accurately based ONLY on the provided PDF content.

Rules:
1. Use ONLY information from the PDF content provided
2. Cite page numbers when you see [PAGE X] markers
3. Be specific and include relevant details
4. If the answer isn't in the content, clearly state that
5. Structure answers clearly with proper formatting
6. Never make up information not in the document"""

                user_prompt = f"""PDF Document: {self.current_pdf}

Content:
{context}

Question: {question}

Provide a detailed, accurate answer based solely on the content above."""

                print("🤖 Generating answer...")

                response = self.client.chat.complete(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1500
                )

                answer = response.choices[0].message.content
                return answer

            except Exception as e:
                error_str = str(e)

                if "rate_limit" in error_str.lower() or "429" in error_str:
                    if attempt < max_retries:
                        wait_time = (attempt + 1) * 5
                        print(f"⏳ Rate limit hit. Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    return "❌ Rate limit exceeded. Please try again in a moment."

                if attempt < max_retries:
                    print(f"⚠️  API error, retrying... (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(2)
                    continue

                return f"❌ Error generating answer: {error_str}"

        return "❌ Maximum retries exceeded. Please try again."

    def get_summary(self) -> str:
        """Generate a concise summary of the loaded PDF."""
        if not self.current_pdf:
            return "❌ No PDF loaded."

        try:
            print("📋 Generating document summary...")

            context_chunks = [chunk for chunk, _ in self.chunks[:3]]
            if len(self.chunks) > 4:
                context_chunks.append(self.chunks[-1][0])

            context = "\n\n".join(context_chunks)

            if len(context) > 15000:
                context = context[:15000]

            prompt = f"""Provide a concise summary of this PDF document in 4-6 sentences. 
Focus on the main topic, key points, and purpose.

Content:
{context}

Summary:"""

            response = self.client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"❌ Error generating summary: {str(e)}"

    def interactive_mode(self):
        """Run interactive Q&A mode with automatic PDF loading only."""
        print("\n" + "=" * 75)
        print("🤖 PDF Question Answering Agent - Powered by Mistral AI")
        print("=" * 75)

        # Auto-load PDF without any manual selection
        if not self.current_pdf:
            if self.available_pdfs:
                print(f"\n✨ Auto-loading PDF from knowledge folder...")
                success = self.load_first_available_pdf()
                if not success:
                    print("❌ Failed to load PDF automatically.")
                    print("💡 Possible solutions:")
                    print("   - Ensure your PDF contains extractable text")
                    print("   - Try a different PDF file")
                    print("   - Check if the PDF is password protected")
                    return
            else:
                print("\n❌ No PDFs found in knowledge folder!")
                print(f"   Please add PDF files to: {self.knowledge_folder}")
                return

        if not self.current_pdf:
            print("❌ No PDF loaded. Exiting.")
            return

        print(f"\n✅ Current PDF: {self.current_pdf}")
        print(f"   📊 {self.pdf_metadata['total_words']:,} words in {self.pdf_metadata['num_chunks']} chunks")
        print("\n" + "=" * 75)
        print("📖 Available Commands:")
        print("=" * 75)
        print("  💬 Ask any question about the PDF content")
        print("  📋 'summary'  - Get a document summary")
        print("  📚 'list'     - List all available PDFs")
        print("  ℹ️  'info'     - Show PDF information")
        print("  🔄 'reload'   - Reload the current PDF")
        print("  🚪 'exit'     - Quit the application")
        print("=" * 75 + "\n")

        question_count = 0

        while True:
            try:
                user_input = input("❓ Your question: ").strip()

                if not user_input:
                    continue

                cmd = user_input.lower()

                if cmd in ['exit', 'quit', 'q', 'bye']:
                    print("\n👋 Thank you for using PDF Mistral Agent. Goodbye!")
                    break

                elif cmd == 'summary':
                    print()
                    summary = self.get_summary()
                    print(f"💡 Summary:\n{summary}")

                elif cmd in ['list', 'pdfs']:
                    self._list_pdfs()

                elif cmd == 'reload':
                    print(f"\n🔄 Reloading {self.current_pdf}...")
                    current_path = self.pdf_metadata.get('path')
                    if current_path:
                        success = self.load_pdf(current_path)
                        if success:
                            print("✅ PDF reloaded successfully!")
                        else:
                            print("❌ Failed to reload PDF")

                elif cmd in ['info', 'details', 'metadata']:
                    print(f"\n📊 PDF Information:")
                    print("=" * 50)
                    for key, value in self.pdf_metadata.items():
                        formatted_key = key.replace('_', ' ').title()
                        print(f"   {formatted_key:20}: {value}")
                    print("=" * 50)

                elif cmd in ['help', 'commands', '?']:
                    print("\n📖 Available Commands:")
                    print("   summary  - Get document summary")
                    print("   list     - List all PDFs")
                    print("   info     - Show PDF details")
                    print("   reload   - Reload current PDF")
                    print("   exit     - Quit application")
                    print("   Or just ask any question!")

                else:
                    question_count += 1
                    print(f"\n🔍 Processing question #{question_count}...")

                    answer = self.ask(user_input)
                    print(f"\n💡 Answer:\n{answer}")

                print("\n" + "-" * 75 + "\n")

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Exiting...")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {str(e)}")
                print("Please try again or type 'exit' to quit.\n")

        if question_count > 0:
            print(f"\n📊 Session Summary:")
            print(f"   Questions answered: {question_count}")
            print(f"   PDF processed: {self.current_pdf}")
        print()

    def _list_pdfs(self):
        """List all available PDFs."""
        self._discover_pdfs()

        if not self.available_pdfs:
            print("\n❌ No PDFs found in knowledge folder.")
            print(f"   Folder location: {self.knowledge_folder}")
        else:
            print(f"\n📚 Available PDFs ({len(self.available_pdfs)}):")
            for pdf_name in self.available_pdfs:
                marker = "✓" if pdf_name == self.current_pdf else " "
                print(f"   {marker} {pdf_name}")


def main():
    """Main entry point with error handling."""
    print("=" * 75)
    print("🚀 PDF Mistral Agent - Starting...")
    print("=" * 75)

    try:
        agent = PDFMistralAgent(
            knowledge_folder="knowledge",
            chunk_size=2500,
            chunk_overlap=300,
            extraction_model="pixtral-large"
        )

        agent.interactive_mode()

    except ValueError as e:
        print(f"\n{str(e)}")
        print("\n💡 Setup Instructions:")
        print("   1. Create a .env file in the project directory")
        print("   2. Add your Mistral API key: MISTRAL_API_KEY=your_key_here")
        print("   3. Get your key from: https://console.mistral.ai/")
        return 1

    except KeyboardInterrupt:
        print("\n👋 Interrupted during startup. Exiting...")
        return 0

    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())