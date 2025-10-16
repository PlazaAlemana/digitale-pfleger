def _extract_text_with_pixtral_large(self, pdf_path: str) -> Tuple[str, Dict]:
    """
    Extract content from complex multi-column PDFs using Pixtral Large.
    Returns both extracted text and extraction data.
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
        extraction_data = {
            "pdf_name": os.path.basename(pdf_path),
            "extraction_date": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_pages": len(images),
            "model": self.vision_model,
            "pages": []
        }

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

            # Store page data for JSON
            extraction_data["pages"].append({
                "page_number": page_num,
                "extracted_text": page_text,
                "character_count": len(page_text),
                "word_count": len(page_text.split())
            })

            # Small delay to avoid rate limits
            if page_num < len(images):
                time.sleep(1)

        extracted_text = "\n\n".join(all_extracted_text)

        if not extracted_text or not extracted_text.strip():
            raise Exception("No text extracted from Pixtral response")

        # Add summary to extraction data
        extraction_data["total_characters"] = len(extracted_text)
        extraction_data["total_words"] = len(extracted_text.split())
        extraction_data["full_text"] = extracted_text

        # Save to JSON if enabled
        if self.save_extraction_json:
            self._save_extraction_json(pdf_path, extraction_data)

        print(f"   ✅ Pixtral Large extraction completed!")
        print(f"   📊 Extracted {len(extracted_text)} characters from {len(images)} pages")

        return extracted_text.strip(), extraction_data

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