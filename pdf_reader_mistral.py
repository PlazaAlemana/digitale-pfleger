import os
import re
import sys
import time
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


class PDFMistralAgent:
    """
    AI Agent that reads PDF documents and answers questions using Mistral AI.
    Handles large PDFs with intelligent chunking and context retrieval.
    Automatically loads PDFs from the 'knowledge' folder.
    """

    def __init__(self, knowledge_folder: str = "knowledge", chunk_size: int = 2500, chunk_overlap: int = 300):
        """
        Initialize the PDF Mistral Agent.

        Args:
            knowledge_folder: Folder containing PDF files (default: "knowledge")
            chunk_size: Size of text chunks (default: 2500 chars)
            chunk_overlap: Overlap between chunks (default: 300 chars)
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

        # Set knowledge folder path (relative to project root)
        self.knowledge_folder = self._get_knowledge_folder_path(knowledge_folder)

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

        # Discover available PDFs
        self._discover_pdfs()

    def _get_knowledge_folder_path(self, folder_name: str) -> str:
        """
        Get the absolute path to the knowledge folder.

        Args:
            folder_name: Name of the knowledge folder

        Returns:
            Absolute path to the knowledge folder
        """
        # Get the script's directory
        script_dir = Path(__file__).parent.absolute()

        # Try to find knowledge folder in project root
        knowledge_path = script_dir / folder_name

        if not knowledge_path.exists():
            # Try parent directory (in case script is in a subfolder)
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

            # Find all PDF files
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
        # Add .pdf extension if not present
        if not filename.lower().endswith('.pdf'):
            filename += '.pdf'

        # Build full path
        pdf_path = os.path.join(self.knowledge_folder, filename)

        return self.load_pdf(pdf_path)

    def load_pdf_by_index(self, index: int) -> bool:
        """
        Load a PDF by its index in the available PDFs list.

        Args:
            index: Index of the PDF (1-based)

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.available_pdfs:
            print("❌ No PDFs available in knowledge folder.")
            return False

        if index < 1 or index > len(self.available_pdfs):
            print(f"❌ Invalid index. Please choose between 1 and {len(self.available_pdfs)}")
            return False

        filename = self.available_pdfs[index - 1]
        return self.load_pdf_by_name(filename)

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
        return self.load_pdf_by_index(1)

    def load_pdf(self, pdf_path: str) -> bool:
        """
        Load and process a PDF file with validation.

        Args:
            pdf_path: Path to the PDF file (absolute or relative)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Clean and normalize path
            pdf_path = pdf_path.strip().strip('"').strip("'")

            # If it's not an absolute path, try relative to knowledge folder
            if not os.path.isabs(pdf_path):
                # Check if it exists as-is first
                if not os.path.exists(pdf_path):
                    # Try in knowledge folder
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

            # Extract text from PDF
            self.pdf_content = self._extract_pdf_text(pdf_path)

            if not self.pdf_content.strip():
                print("❌ Error: PDF appears to be empty or contains no extractable text")
                print("   (It might be an image-based PDF that requires OCR)")
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

    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF with page numbers and progress indication."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)

                print(f"   📖 Extracting text from {total_pages} pages...")

                all_text = []
                show_progress = total_pages > 20

                for page_num, page in enumerate(pdf_reader.pages, start=1):
                    # Show progress for large PDFs
                    if show_progress and page_num % 10 == 0:
                        print(f"      Progress: {page_num}/{total_pages} pages...")

                    try:
                        text = page.extract_text() or ""
                        if text.strip():
                            # Add page marker for reference
                            text = f"[PAGE {page_num}]\n{text}"
                            all_text.append(text)
                    except Exception as e:
                        print(f"      ⚠️  Warning: Could not extract page {page_num}")
                        continue

                if not all_text:
                    raise Exception("No text could be extracted from any page")

                # Join all pages
                full_content = "\n\n".join(all_text)

                # Clean up excessive whitespace while preserving structure
                full_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', full_content)
                full_content = re.sub(r'[ \t]+', ' ', full_content)

                return full_content.strip()

        except Exception as e:
            raise Exception(f"Failed to extract PDF text: {str(e)}")

    def _create_chunks(self, text: str) -> List[Tuple[str, int]]:
        """
        Split text into overlapping chunks with position tracking.

        Args:
            text: Full text to chunk

        Returns:
            List of (chunk_text, start_position) tuples
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size

            # If not the last chunk, try to break at sentence or word boundary
            if end < text_length:
                # Look for sentence end (., !, ?)
                for punct in ['. ', '! ', '? ']:
                    sentence_end = text.rfind(punct, start + self.chunk_size // 2, end)
                    if sentence_end > start:
                        end = sentence_end + 1
                        break
                else:
                    # Look for paragraph break
                    para_break = text.rfind('\n\n', start, end)
                    if para_break > start + self.chunk_size // 2:
                        end = para_break + 2
                    else:
                        # Look for any newline
                        newline = text.rfind('\n', start, end)
                        if newline > start:
                            end = newline
                        else:
                            # Last resort: word boundary
                            space = text.rfind(' ', start, end)
                            if space > start:
                                end = space

            chunk = text[start:end].strip()
            if chunk:
                chunks.append((chunk, start))

            # Move start position with overlap
            start = end - self.chunk_overlap

            # Prevent infinite loop
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

        # Extract meaningful keywords (4+ chars, not common words)
        common_words = {'what', 'where', 'when', 'which', 'who', 'how', 'does',
                        'this', 'that', 'these', 'those', 'with', 'from', 'about',
                        'have', 'been', 'were', 'their', 'there', 'would', 'could'}
        keywords = [w.lower() for w in re.findall(r'\b\w{4,}\b', question.lower())
                    if w.lower() not in common_words]

        if not keywords:
            # If no good keywords, use first chunks
            return [chunk for chunk, _ in self.chunks[:top_k]]

        # Score chunks
        chunk_scores = []
        for i, (chunk, position) in enumerate(self.chunks):
            chunk_lower = chunk.lower()

            # Keyword frequency score
            keyword_score = sum(chunk_lower.count(kw) for kw in keywords)

            # Bonus for exact phrase match
            phrase_bonus = 10 if any(kw in chunk_lower for kw in keywords) else 0

            # Slight bonus for earlier chunks (often contain important context)
            position_bonus = max(0, 5 - (i / len(self.chunks)) * 5)

            total_score = keyword_score + phrase_bonus + position_bonus
            chunk_scores.append((total_score, i, chunk))

        # Sort by score
        chunk_scores.sort(reverse=True, key=lambda x: x[0])

        # Get top k chunks
        top_chunks = [chunk for _, _, chunk in chunk_scores[:top_k]]

        # If no good matches, return first chunks
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
                # Determine context to use
                if use_full_context or len(self.pdf_content) < 8000:
                    context = self.pdf_content
                    print("🔍 Using full PDF context...")
                else:
                    relevant_chunks = self._find_relevant_chunks(question, top_k=4)
                    context = "\n\n---\n\n".join(relevant_chunks)
                    print(f"🔍 Using {len(relevant_chunks)} relevant chunks...")

                # Truncate if too long (Mistral context limits)
                max_context_length = 25000
                if len(context) > max_context_length:
                    context = context[:max_context_length] + "\n\n[... content truncated ...]"
                    print(f"   ⚠️  Context truncated to {max_context_length} chars")

                # Build prompt
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

                # Call Mistral API
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

                # Handle rate limiting
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    if attempt < max_retries:
                        wait_time = (attempt + 1) * 5
                        print(f"⏳ Rate limit hit. Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    return "❌ Rate limit exceeded. Please try again in a moment."

                # Handle other API errors
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

            # Use first chunks and last chunk for summary
            context_chunks = [chunk for chunk, _ in self.chunks[:3]]
            if len(self.chunks) > 4:
                context_chunks.append(self.chunks[-1][0])

            context = "\n\n".join(context_chunks)

            # Truncate if needed
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
        """Run interactive Q&A mode with enhanced UX and automatic PDF loading."""
        print("\n" + "=" * 75)
        print("🤖 PDF Question Answering Agent - Powered by Mistral AI")
        print("=" * 75)

        # Auto-load PDF if available
        if not self.current_pdf:
            if self.available_pdfs:
                if len(self.available_pdfs) == 1:
                    # Auto-load if only one PDF
                    print("\n✨ Auto-loading PDF from knowledge folder...")
                    if not self.load_first_available_pdf():
                        print("❌ Failed to auto-load PDF. Please select manually.")
                        self._manual_pdf_selection()
                else:
                    # Let user choose
                    self._manual_pdf_selection()
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
        print("  📂 'switch'   - Switch to a different PDF")
        print("  📚 'list'     - List all available PDFs")
        print("  ℹ️  'info'     - Show PDF information")
        print("  🔄 'reload'   - Reload the current PDF")
        print("  🚪 'exit'     - Quit the application")
        print("=" * 75 + "\n")

        question_count = 0

        while True:
            try:
                user_input = input("❓ Your input: ").strip()

                if not user_input:
                    continue

                # Handle commands
                cmd = user_input.lower()

                if cmd in ['exit', 'quit', 'q', 'bye']:
                    print("\n👋 Thank you for using PDF Mistral Agent. Goodbye!")
                    break

                elif cmd == 'summary':
                    print()
                    summary = self.get_summary()
                    print(f"💡 Summary:\n{summary}")

                elif cmd in ['switch', 'change', 'load']:
                    self._manual_pdf_selection()
                    if self.current_pdf:
                        question_count = 0  # Reset counter

                elif cmd in ['list', 'pdfs']:
                    self._list_pdfs()

                elif cmd == 'reload':
                    print(f"\n🔄 Reloading {self.current_pdf}...")
                    current_path = self.pdf_metadata.get('path')
                    if current_path:
                        self.load_pdf(current_path)

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
                    print("   switch   - Switch to different PDF")
                    print("   list     - List all PDFs")
                    print("   info     - Show PDF details")
                    print("   reload   - Reload current PDF")
                    print("   exit     - Quit application")
                    print("   Or just ask any question!")

                else:
                    # Regular question
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

    def _manual_pdf_selection(self):
        """Allow user to manually select a PDF from available options."""
        if not self.available_pdfs:
            print("\n❌ No PDFs available in knowledge folder.")
            return

        print("\n📚 Available PDFs:")
        for i, pdf_name in enumerate(self.available_pdfs, 1):
            print(f"   {i}. {pdf_name}")

        print("\nSelect a PDF by number, or press Enter to cancel:")
        choice = input("Selection: ").strip()

        if not choice:
            return

        try:
            index = int(choice)
            self.load_pdf_by_index(index)
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

    def _list_pdfs(self):
        """List all available PDFs."""
        self._discover_pdfs()

        if not self.available_pdfs:
            print("\n❌ No PDFs found in knowledge folder.")
            print(f"   Folder location: {self.knowledge_folder}")
        else:
            print(f"\n📚 Available PDFs ({len(self.available_pdfs)}):")
            for i, pdf_name in enumerate(self.available_pdfs, 1):
                marker = "✓" if pdf_name == self.current_pdf else " "
                print(f"   {marker} {i}. {pdf_name}")


def main():
    """Main entry point with error handling."""
    print("=" * 75)
    print("🚀 PDF Mistral Agent - Starting...")
    print("=" * 75)

    try:
        # Create agent instance with knowledge folder
        agent = PDFMistralAgent(
            knowledge_folder="knowledge",
            chunk_size=2500,
            chunk_overlap=300
        )

        # Run interactive mode
        agent.interactive_mode()

    except ValueError as e:
        print(f"\n{str(e)}")
        print("\n💡 Setup Instructions:")
        print("   1. Create a .env file in the project directory")
        print("   2. Add your Mistral API key: MISTRAL_API_KEY=your_key_here")
        print("   3. Get your key from: https://console.mistral.ai/")
        return 1

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted during startup. Exiting...")
        return 0

    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())