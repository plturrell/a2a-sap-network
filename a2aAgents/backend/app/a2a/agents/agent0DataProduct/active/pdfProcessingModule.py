import asyncio
import io
import logging
import os
import tempfile
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from datetime import datetime
import aiofiles
from asyncio_throttle import Throttler

from app.a2a.core.security_base import SecureA2AAgent
"""
Enhanced PDF Processing Module for Agent 0
Implements comprehensive PDF processing capabilities with MCP skills architecture
"""

# Core PDF processing libraries
try:
    import PyPDF2
    import pdfplumber
    import tabula
    import fitz  # PyMuPDF
    import camelot
    from pdfminer.high_level import extract_text
    from pikepdf import Pdf
    PDF_LIBRARIES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PDF libraries not available: {e}")
    PDF_LIBRARIES_AVAILABLE = False

# OCR libraries
try:
    import pytesseract
    import cv2
    from pdf2image import convert_from_path, convert_from_bytes
    OCR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"OCR libraries not available: {e}")
    OCR_AVAILABLE = False

# AWS Textract (optional)
try:
    import boto3
    AWS_TEXTRACT_AVAILABLE = True
except ImportError:
    AWS_TEXTRACT_AVAILABLE = False

# Grok AI client for enhanced analysis
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'reasoningAgent'))
    from asyncGrokClient import AsyncGrokReasoning, GrokConfig
    GROK_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Grok client not available: {e}")
    GROK_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnhancedPDFProcessor(SecureA2AAgent):
    """Enhanced PDF processor with streaming, OCR, and advanced extraction"""

    # Security features provided by SecureA2AAgent:
    # - JWT authentication and authorization
    # - Rate limiting and request throttling
    # - Input validation and sanitization
    # - Audit logging and compliance tracking
    # - Encrypted communication channels
    # - Automatic security scanning

    def __init__(self, max_concurrent_files: int = 5, chunk_size: int = 1024*1024):

        super().__init__()
        self.throttler = Throttler(rate_limit=max_concurrent_files)
        self.chunk_size = chunk_size
        self.temp_dir = tempfile.mkdtemp(prefix="a2a_pdf_")

        # Initialize Grok client for AI-powered analysis
        self.grok_client = None
        if GROK_AVAILABLE:
            try:
                grok_config = GrokConfig(
                    api_key=os.getenv('XAI_API_KEY') or os.getenv('GROK_API_KEY'),
                    cache_ttl=600,  # 10 minutes cache for PDF analysis
                    pool_connections=3
                )
                self.grok_client = AsyncGrokReasoning(grok_config)
                logger.info("Grok client initialized for PDF analysis")
            except Exception as e:
                logger.warning(f"Failed to initialize Grok client: {e}")
                self.grok_client = None

    async def extract_pdf_tables(self, pdf_path: str, use_ocr: bool = False) -> List[Dict[str, Any]]:
        """Extract tables from PDF with multiple methods"""
        if not PDF_LIBRARIES_AVAILABLE:
            raise RuntimeError("PDF libraries not available")

        tables = []

        try:
            # Method 1: Camelot (best for well-formatted tables)
            camelot_tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            for i, table in enumerate(camelot_tables):
                tables.append({
                    "method": "camelot_lattice",
                    "page": table.page,
                    "table_index": i,
                    "data": table.df.to_dict('records'),
                    "accuracy": table.accuracy,
                    "whitespace": table.whitespace
                })

            # Method 2: Tabula (good for stream tables)
            if not tables:  # Fallback if camelot fails
                tabula_tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
                for i, df in enumerate(tabula_tables):
                    tables.append({
                        "method": "tabula",
                        "page": i + 1,
                        "table_index": i,
                        "data": df.to_dict('records'),
                        "accuracy": 0.8,  # Default estimate
                        "whitespace": 0.1
                    })

            # Method 3: OCR-based extraction if enabled
            if use_ocr and OCR_AVAILABLE and not tables:
                ocr_tables = await self._extract_tables_ocr(pdf_path)
                tables.extend(ocr_tables)

        except Exception as e:
            logger.error(f"Table extraction failed: {e}")

        return tables

    async def extract_pdf_text(self, pdf_path: str, use_ocr: bool = False) -> str:
        """Extract text from PDF with fallback methods"""
        if not PDF_LIBRARIES_AVAILABLE:
            raise RuntimeError("PDF libraries not available")

        text_content = ""

        try:
            # Method 1: PyMuPDF (fastest)
            doc = fitz.open(pdf_path)
            for page in doc:
                text_content += page.get_text()
            doc.close()

            # Method 2: pdfplumber (better formatting)
            if not text_content.strip():
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text + "\n"

            # Method 3: OCR fallback
            if use_ocr and OCR_AVAILABLE and not text_content.strip():
                text_content = await self._extract_text_ocr(pdf_path)

        except Exception as e:
            logger.error(f"Text extraction failed: {e}")

        return text_content

    async def extract_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract comprehensive PDF metadata"""
        if not PDF_LIBRARIES_AVAILABLE:
            raise RuntimeError("PDF libraries not available")

        metadata = {
            "file_info": {},
            "document_info": {},
            "structure_info": {},
            "security_info": {}
        }

        try:
            # File-level metadata
            file_stats = os.stat(pdf_path)
            metadata["file_info"] = {
                "file_size": file_stats.st_size,
                "created": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                "file_path": pdf_path
            }

            # Document metadata using PyMuPDF
            doc = fitz.open(pdf_path)
            doc_metadata = doc.metadata
            metadata["document_info"] = {
                "title": doc_metadata.get("title", ""),
                "author": doc_metadata.get("author", ""),
                "subject": doc_metadata.get("subject", ""),
                "creator": doc_metadata.get("creator", ""),
                "producer": doc_metadata.get("producer", ""),
                "creation_date": doc_metadata.get("creationDate", ""),
                "modification_date": doc_metadata.get("modDate", "")
            }

            # Structure information
            metadata["structure_info"] = {
                "page_count": doc.page_count,
                "has_links": any(page.get_links() for page in doc),
                "has_annotations": any(page.annots() for page in doc),
                "has_forms": doc.is_form_pdf,
                "is_encrypted": doc.needs_pass,
                "pdf_version": doc.pdf_version()
            }

            # Security information
            metadata["security_info"] = {
                "is_encrypted": doc.needs_pass,
                "permissions": doc.permissions if hasattr(doc, 'permissions') else {},
                "is_linearized": doc.is_pdf and hasattr(doc, 'is_fast_web_view')
            }

            doc.close()

        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")

        return metadata

    async def extract_pdf_forms(self, pdf_path: str) -> Dict[str, Any]:
        """Extract PDF form fields and data"""
        if not PDF_LIBRARIES_AVAILABLE:
            raise RuntimeError("PDF libraries not available")

        form_data = {
            "has_forms": False,
            "form_fields": [],
            "filled_fields": {},
            "form_structure": {}
        }

        try:
            # Using PyMuPDF for form extraction
            doc = fitz.open(pdf_path)

            if doc.is_form_pdf:
                form_data["has_forms"] = True

                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    widgets = page.widgets()

                    for widget in widgets:
                        field_info = {
                            "field_name": widget.field_name,
                            "field_type": widget.field_type_string,
                            "field_value": widget.field_value,
                            "page": page_num + 1,
                            "rect": list(widget.rect),
                            "is_required": widget.field_flags & 2 != 0
                        }
                        form_data["form_fields"].append(field_info)

                        if widget.field_value:
                            form_data["filled_fields"][widget.field_name] = widget.field_value

            doc.close()

        except Exception as e:
            logger.error(f"Form extraction failed: {e}")

        return form_data

    async def stream_large_pdf(self, pdf_path: str, chunk_pages: int = 10) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream processing for large PDF files"""
        if not PDF_LIBRARIES_AVAILABLE:
            raise RuntimeError("PDF libraries not available")

        try:
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count

            for start_page in range(0, total_pages, chunk_pages):
                end_page = min(start_page + chunk_pages, total_pages)

                chunk_data = {
                    "chunk_info": {
                        "start_page": start_page + 1,
                        "end_page": end_page,
                        "total_pages": total_pages,
                        "chunk_size": end_page - start_page
                    },
                    "text_content": "",
                    "tables": [],
                    "images": [],
                    "metadata": {}
                }

                # Extract content from chunk
                for page_num in range(start_page, end_page):
                    page = doc[page_num]
                    chunk_data["text_content"] += page.get_text()

                    # Extract images
                    image_list = page.get_images()
                    for img_index, img in enumerate(image_list):
                        chunk_data["images"].append({
                            "page": page_num + 1,
                            "image_index": img_index,
                            "xref": img[0],
                            "width": img[2],
                            "height": img[3]
                        })

                yield chunk_data

                # Throttle to prevent memory issues
                await asyncio.sleep(0.1)

            doc.close()

        except Exception as e:
            logger.error(f"PDF streaming failed: {e}")
            yield {"error": str(e)}

    async def _extract_tables_ocr(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables using OCR"""
        if not OCR_AVAILABLE:
            return []

        tables = []
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path)

            for page_num, image in enumerate(images):
                # Use Tesseract to extract text with table structure
                custom_config = r'--oem 3 --psm 6 -c tessedit_create_tsv=1'
                tsv_data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)

                # Process TSV data to identify table structures
                table_data = self._process_ocr_table_data(tsv_data)
                if table_data:
                    tables.append({
                        "method": "ocr_tesseract",
                        "page": page_num + 1,
                        "table_index": 0,
                        "data": table_data,
                        "accuracy": 0.7,  # OCR typically less accurate
                        "whitespace": 0.2
                    })

        except Exception as e:
            logger.error(f"OCR table extraction failed: {e}")

        return tables

    async def _extract_text_ocr(self, pdf_path: str) -> str:
        """Extract text using OCR"""
        if not OCR_AVAILABLE:
            return ""

        text_content = ""
        try:
            images = convert_from_path(pdf_path)

            for image in images:
                text_content += pytesseract.image_to_string(image) + "\n"

        except Exception as e:
            logger.error(f"OCR text extraction failed: {e}")

        return text_content

    async def analyze_pdf_content_with_grok(self, text_content: str, tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze PDF content using Grok AI for enhanced insights"""
        if not self.grok_client:
            return {"analysis": "Grok client not available", "insights": [], "quality_score": 0.5}

        try:
            # Prepare content summary for analysis
            content_summary = {
                "text_length": len(text_content),
                "table_count": len(tables),
                "text_preview": text_content[:1000] if text_content else "",
                "table_preview": tables[:2] if tables else []
            }

            # Use Grok to analyze content structure and quality
            analysis_prompt = f"""
            Analyze this PDF content and provide insights:

            Content Summary: {content_summary}

            Provide analysis including:
            1. Content type and structure assessment
            2. Data quality indicators
            3. Key insights and patterns
            4. Extraction completeness score (0-1)
            5. Recommended processing improvements

            Return as JSON with fields: content_type, quality_score, insights, recommendations
            """

            result = await self.grok_client.decompose_question(analysis_prompt)

            if result.get("success"):
                decomposition = result.get("decomposition", {})
                return {
                    "analysis": "Grok AI analysis completed",
                    "content_type": decomposition.get("content_type", "unknown"),
                    "quality_score": decomposition.get("quality_score", 0.7),
                    "insights": decomposition.get("insights", []),
                    "recommendations": decomposition.get("recommendations", []),
                    "grok_cached": result.get("cached", False),
                    "response_time": result.get("response_time", 0)
                }
            else:
                return {"analysis": "Grok analysis failed", "quality_score": 0.5}

        except Exception as e:
            logger.error(f"Grok PDF analysis failed: {e}")
            return {"analysis": f"Analysis error: {e}", "quality_score": 0.5}

    def _process_ocr_table_data(self, tsv_data: Dict) -> List[Dict[str, Any]]:
        """Process OCR TSV data to extract table structure"""
        rows = {}

        for i, text in enumerate(tsv_data['text']):
            if text.strip():
                top = tsv_data['top'][i]
                left = tsv_data['left'][i]

                # Group by approximate row (top position)
                row_key = round(top / 20) * 20  # Group within 20 pixels

                if row_key not in rows:
                    rows[row_key] = []

                rows[row_key].append({
                    'text': text,
                    'left': left,
                    'confidence': tsv_data['conf'][i]
                })

        # Convert to table format
        table_data = []
        for row_key in sorted(rows.keys()):
            row_cells = sorted(rows[row_key], key=lambda x: x['left'])
            row_data = {f"col_{i}": cell['text'] for i, cell in enumerate(row_cells)}
            table_data.append(row_data)

        return table_data

    async def cleanup(self):
        """Cleanup temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
