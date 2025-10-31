
# services/extractors/grobid_extractor.py
import httpx
import xml.etree.ElementTree as ET
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import asyncio
from loguru import logger

from app.models.schemas import (
    Metadata, Section, Reference, Author, 
    Paragraph, SectionType, BoundingBox
)
from app.config import settings
from app.utils.exceptions import ExtractionError


class GROBIDExtractor:
    """
    GROBID service integration for extracting structured content from academic PDFs.
    Handles: metadata, sections, paragraphs, references
    """
    
    NAMESPACES = {
        'tei': 'http://www.tei-c.org/ns/1.0'
    }
    
    SECTION_MAPPING = {
        'abstract': SectionType.ABSTRACT,
        'introduction': SectionType.INTRODUCTION,
        'related': SectionType.RELATED_WORK,
        'method': SectionType.METHODOLOGY,
        'experiment': SectionType.EXPERIMENTS,
        'result': SectionType.RESULTS,
        'discussion': SectionType.DISCUSSION,
        'conclusion': SectionType.CONCLUSION,
        'appendix': SectionType.APPENDIX,
    }
    
    # Enhanced section mapping with more patterns
    ENHANCED_SECTION_MAPPING = {
        # Abstract and introduction
        'abstract': SectionType.ABSTRACT,
        'summary': SectionType.ABSTRACT,
        'introduction': SectionType.INTRODUCTION,
        'intro': SectionType.INTRODUCTION,
        'background': SectionType.INTRODUCTION,
        
        # Related work
        'related work': SectionType.RELATED_WORK,
        'related': SectionType.RELATED_WORK,
        'literature review': SectionType.RELATED_WORK,
        'previous work': SectionType.RELATED_WORK,
        'state of the art': SectionType.RELATED_WORK,
        
        # Methodology
        'method': SectionType.METHODOLOGY,
        'methodology': SectionType.METHODOLOGY,
        'methods': SectionType.METHODOLOGY,
        'approach': SectionType.METHODOLOGY,
        'algorithm': SectionType.METHODOLOGY,
        'design': SectionType.METHODOLOGY,
        
        # Experiments and implementation
        'experiment': SectionType.EXPERIMENTS,
        'experiments': SectionType.EXPERIMENTS,
        'implementation': SectionType.EXPERIMENTS,
        'evaluation': SectionType.EXPERIMENTS,
        'setup': SectionType.EXPERIMENTS,
        
        # Results
        'result': SectionType.RESULTS,
        'results': SectionType.RESULTS,
        'evaluation results': SectionType.RESULTS,
        'performance': SectionType.RESULTS,
        
        # Discussion
        'discussion': SectionType.DISCUSSION,
        'analysis': SectionType.DISCUSSION,
        'interpretation': SectionType.DISCUSSION,
        
        # Conclusion
        'conclusion': SectionType.CONCLUSION,
        'conclusions': SectionType.CONCLUSION,
        'summary': SectionType.CONCLUSION,
        'future work': SectionType.CONCLUSION,
        
        # Appendix
        'appendix': SectionType.APPENDIX,
        'appendices': SectionType.APPENDIX,
    }
    
    def __init__(self, grobid_url: str = None):
        self.grobid_url = grobid_url or settings.grobid_url
        self.client = None  # Initialize client lazily
        self._service_available = None  # Cache service availability
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def _ensure_client(self):
        """Ensure HTTP client is initialized"""
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=180.0)
    
    async def check_service(self) -> bool:
        """Check if GROBID service is available"""
        if self._service_available is not None:
            return self._service_available
            
        try:
            await self._ensure_client()
            response = await self.client.get(f"{self.grobid_url}/api/isalive")
            self._service_available = response.status_code == 200
            return self._service_available
        except Exception as e:
            logger.warning(f"GROBID service unavailable: {e}")
            self._service_available = False
            return False
    
    async def extract(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract structured content from PDF using GROBID
        
        Returns:
            Dictionary containing metadata, sections, and references
        """
        # Get page count from PDF
        page_count = self._get_pdf_page_count(pdf_path)
        
        if not await self.check_service():
            # Return empty result instead of raising error
            logger.warning("GROBID service not available, returning empty result")
            return {
                'metadata': Metadata(title="Unknown", page_count=page_count),
                'sections': [],
                'references': [],
                'raw_tei': None
            }
        
        try:
            # Store PDF path for page count fallback
            self.pdf_path = pdf_path
            
            # Process full document
            tei_xml = await self._process_fulltext(pdf_path)
            
            # Parse TEI XML
            root = ET.fromstring(tei_xml)
            
            # Extract components
            metadata = self._extract_metadata(root)
            sections = self._extract_sections_enhanced(root)
            references = self._extract_references(root)
            
            return {
                'metadata': metadata,
                'sections': sections,
                'references': references,
                'raw_tei': tei_xml  # Keep for debugging
            }
            
        except Exception as e:
            logger.error(f"GROBID extraction failed: {e}")
            raise ExtractionError(f"GROBID extraction failed: {str(e)}")
    
    async def _process_fulltext(self, pdf_path: Path) -> str:
        """Process PDF with GROBID fulltext endpoint"""
        await self._ensure_client()
        
        with open(pdf_path, 'rb') as f:
            files = {'input': (pdf_path.name, f, 'application/pdf')}
            
            response = await self.client.post(
                f"{self.grobid_url}/api/processFulltextDocument",
                files=files,
                data={
                    'consolidateHeader': '1',
                    'consolidateCitations': '1',
                    'includeRawCitations': '1',
                    'includeRawAffiliations': '1',
                    'teiCoordinates': 'true'
                }
            )
            
            if response.status_code != 200:
                raise ExtractionError(f"GROBID API error: {response.status_code}")
            
            return response.text
    
    def _extract_metadata(self, root: ET.Element) -> Metadata:
        """Extract metadata from TEI header"""
        header = root.find('.//tei:teiHeader', self.NAMESPACES)
        if header is None:
            return Metadata(title="Unknown")
        
        # Title
        title_elem = header.find('.//tei:titleStmt/tei:title', self.NAMESPACES)
        title = title_elem.text if title_elem is not None else "Unknown"
        
        # Authors
        authors = []
        for author_elem in header.findall('.//tei:author', self.NAMESPACES):
            name_parts = []
            forename = author_elem.find('.//tei:forename', self.NAMESPACES)
            surname = author_elem.find('.//tei:surname', self.NAMESPACES)
            
            if forename is not None:
                name_parts.append(forename.text)
            if surname is not None:
                name_parts.append(surname.text)
            
            if name_parts:
                name = ' '.join(name_parts)
                
                # Affiliation
                affiliation = None
                aff_elem = author_elem.find('.//tei:affiliation', self.NAMESPACES)
                if aff_elem is not None:
                    org_name = aff_elem.find('.//tei:orgName', self.NAMESPACES)
                    if org_name is not None:
                        affiliation = org_name.text
                
                # Email
                email = None
                email_elem = author_elem.find('.//tei:email', self.NAMESPACES)
                if email_elem is not None:
                    email = email_elem.text
                
                authors.append(Author(
                    name=name,
                    affiliation=affiliation,
                    email=email
                ))
        
        # Abstract
        abstract = None
        abstract_elem = root.find('.//tei:abstract', self.NAMESPACES)
        if abstract_elem is not None:
            abstract_parts = []
            for p in abstract_elem.findall('.//tei:p', self.NAMESPACES):
                if p.text:
                    abstract_parts.append(p.text.strip())
            abstract = ' '.join(abstract_parts)
        
        # Keywords
        keywords = []
        for kw in root.findall('.//tei:keywords/tei:term', self.NAMESPACES):
            if kw.text:
                keywords.append(kw.text)
        
        # DOI
        doi = None
        idno = header.find('.//tei:idno[@type="DOI"]', self.NAMESPACES)
        if idno is not None:
            doi = idno.text
        
        # Year
        year = None
        date_elem = header.find('.//tei:date', self.NAMESPACES)
        if date_elem is not None and date_elem.get('when'):
            try:
                year = int(date_elem.get('when')[:4])
            except:
                pass
        
        # Get page count from TEI or fallback to PDF
        page_count = self._get_pdf_page_count_from_tei(root)
        if page_count == 0:
            # Fallback to getting page count from PDF file
            page_count = self._get_pdf_page_count(self.pdf_path) if hasattr(self, 'pdf_path') else 0
        
        return Metadata(
            title=title,
            authors=authors,
            abstract=abstract,
            keywords=keywords,
            doi=doi,
            year=year,
            page_count=page_count
        )
    
    def _get_pdf_page_count(self, pdf_path: Path) -> int:
        """Get the number of pages in a PDF file"""
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            page_count = len(doc)
            doc.close()
            return page_count
        except Exception as e:
            logger.warning(f"Failed to get page count for {pdf_path}: {e}")
            return 0
    
    def _get_pdf_page_count_from_tei(self, root: ET.Element) -> int:
        """Extract page count from TEI XML if available"""
        try:
            # Look for page count in TEI header
            header = root.find('.//tei:teiHeader', self.NAMESPACES)
            if header is not None:
                # Check for page count in various possible locations
                page_count_elem = header.find('.//tei:extent/tei:measure[@unit="page"]', self.NAMESPACES)
                if page_count_elem is not None and page_count_elem.text:
                    try:
                        return int(page_count_elem.text)
                    except ValueError:
                        pass
                
                # Alternative: count pages in body
                body = root.find('.//tei:body', self.NAMESPACES)
                if body is not None:
                    pages = body.findall('.//tei:pb', self.NAMESPACES)
                    if pages:
                        return len(pages) + 1  # +1 because first page doesn't have pb element
        except Exception as e:
            logger.warning(f"Failed to extract page count from TEI: {e}")
        
        return 0
    
    def _extract_sections(self, root: ET.Element) -> List[Section]:
        """Extract document sections from TEI body"""
        sections = []
        body = root.find('.//tei:body', self.NAMESPACES)
        
        if body is None:
            return sections
        
        section_id = 0
        for div in body.findall('.//tei:div', self.NAMESPACES):
            section = self._parse_section(div, section_id)
            if section:
                sections.append(section)
                section_id += 1
        
        return sections
    
    def _extract_sections_enhanced(self, root: ET.Element) -> List[Section]:
        """Enhanced section extraction with improved detection and hierarchical structure"""
        sections = []
        body = root.find('.//tei:body', self.NAMESPACES)
        
        if body is None:
            return sections
        
        # Enhanced section detection with hierarchical structure
        sections = self._extract_hierarchical_sections(body)
        
        # Post-process sections for better accuracy
        sections = self._post_process_sections(sections)
        
        return sections
    
    def _extract_hierarchical_sections(self, body: ET.Element) -> List[Section]:
        """Extract sections with hierarchical structure detection"""
        sections = []
        section_id = 0
        
        # Find all div elements (potential sections)
        divs = body.findall('.//tei:div', self.NAMESPACES)
        
        for div in divs:
            section = self._parse_section_enhanced(div, section_id)
            if section:
                sections.append(section)
                section_id += 1
        
        # If no sections found, try alternative parsing
        if not sections:
            sections = self._extract_sections_alternative(body, section_id)
        
        return sections
    
    def _parse_section(self, div: ET.Element, section_id: int) -> Optional[Section]:
        """Parse a single section from TEI div element"""
        # Get section title
        head = div.find('tei:head', self.NAMESPACES)
        if head is None:
            return None
        
        title = head.text or f"Section {section_id + 1}"
        
        # Determine section type
        section_type = SectionType.OTHER
        title_lower = title.lower()
        for keyword, stype in self.SECTION_MAPPING.items():
            if keyword in title_lower:
                section_type = stype
                break
        
        # Extract paragraphs with improved page detection
        paragraphs = []
        page_nums = set()
        
        for p_elem in div.findall('tei:p', self.NAMESPACES):
            text = self._extract_text_from_element(p_elem)
            if text:
                # Use improved page detection
                page = self._get_accurate_page_number(p_elem, text)
                page_nums.add(page)
                paragraphs.append(Paragraph(
                    text=text,
                    page=page
                ))
        
        if not paragraphs:
            return None
        
        # Get label if exists
        label = head.get('n')
        
        return Section(
            label=label,
            title=title,
            type=section_type,
            level=1,  # TODO: detect heading levels
            page_start=min(page_nums) if page_nums else 1,
            page_end=max(page_nums) if page_nums else 1,
            paragraphs=paragraphs
        )
    
    def _parse_section_enhanced(self, div: ET.Element, section_id: int) -> Optional[Section]:
        """Enhanced parsing of a single section from TEI div element"""
        # Get section title
        head = div.find('tei:head', self.NAMESPACES)
        if head is None:
            return None
        
        title = head.text or f"Section {section_id + 1}"
        
        # Determine section type
        section_type = self._determine_section_type_enhanced(title)
        
        # Extract paragraphs with improved page detection
        paragraphs, page_nums = self._extract_paragraphs_enhanced(div)
        
        if not paragraphs:
            return None
        
        # Get label if exists
        label = self._extract_section_label(div)
        
        # Calculate accurate page range with validation
        page_start, page_end = self._calculate_accurate_section_page_range(page_nums, paragraphs)
        
        return Section(
            label=label,
            title=title,
            type=section_type,
            level=self._detect_section_level(div, title),
            page_start=page_start,
            page_end=page_end,
            paragraphs=paragraphs
        )
    
    def _extract_sections_alternative(self, body: ET.Element, section_id: int) -> List[Section]:
        """Alternative section extraction if hierarchical parsing fails"""
        sections = []
        # This is a placeholder. In a real scenario, you'd implement a more robust
        # alternative parsing logic here, e.g., based on headings, bold text, etc.
        # For now, it just returns an empty list.
        return sections
    
    def _post_process_sections(self, sections: List[Section]) -> List[Section]:
        """Post-process sections to ensure correct hierarchical structure and page ranges"""
        # This is a placeholder. In a real scenario, you'd implement logic
        # to merge overlapping sections, adjust page ranges, etc.
        return sections
    
    def _get_accurate_page_number(self, p_elem: ET.Element, text: str) -> int:
        """Get accurate page number using multiple methods"""
        # Method 1: Try to get from coordinates (GROBID's method)
        coords = p_elem.get('coords')
        if coords:
            try:
                # Parse coords format: "page,x1,y1,x2,y2;..."
                page = int(coords.split(',')[0])
                if page > 0:
                    return page
            except:
                pass
        
        # Method 2: Use PyMuPDF to find page by text content
        if hasattr(self, 'pdf_path') and self.pdf_path.exists():
            try:
                import fitz
                doc = fitz.open(str(self.pdf_path))
                
                # Search for the text in each page
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    
                    # Check if this text appears on this page
                    # Use a more flexible matching approach
                    if self._text_matches_page(text, page_text):
                        doc.close()
                        return page_num + 1
                
                doc.close()
            except Exception as e:
                logger.warning(f"Failed to use PyMuPDF for page detection: {e}")
        
        # Method 3: Fallback to page 1
        return 1
    
    def _text_matches_page(self, search_text: str, page_text: str) -> bool:
        """Check if text content matches a page using flexible matching"""
        # Clean and normalize text
        search_clean = self._normalize_text(search_text)
        page_clean = self._normalize_text(page_text)
        
        # If search text is very short, require exact match
        if len(search_clean) < 20:
            return search_clean in page_clean
        
        # For longer text, use partial matching
        # Split into words and check if most words are present
        search_words = set(search_clean.split())
        page_words = set(page_clean.split())
        
        if not search_words:
            return False
        
        # Calculate word overlap
        common_words = search_words.intersection(page_words)
        overlap_ratio = len(common_words) / len(search_words)
        
        # Require at least 70% word overlap for a match
        return overlap_ratio >= 0.7
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        import re
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Convert to lowercase for comparison
        text = text.lower()
        # Remove punctuation for better matching
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def _extract_text_from_element(self, elem: ET.Element) -> str:
        """Extract all text from an element and its children"""
        texts = []
        
        if elem.text:
            texts.append(elem.text)
        
        for child in elem:
            if child.tag.endswith('ref'):
                # Handle references
                if child.text:
                    texts.append(child.text)
            else:
                # Recursively extract text
                child_text = self._extract_text_from_element(child)
                if child_text:
                    texts.append(child_text)
            
            if child.tail:
                texts.append(child.tail)
        
        return ' '.join(texts).strip()
    
    def _extract_references(self, root: ET.Element) -> List[Reference]:
        """Extract bibliographic references"""
        references = []
        
        # Find bibliography section
        bibl_struct_list = root.findall('.//tei:listBibl/tei:biblStruct', self.NAMESPACES)
        
        for bibl in bibl_struct_list:
            ref = self._parse_reference(bibl)
            if ref:
                references.append(ref)
        
        return references
    
    def _parse_reference(self, bibl: ET.Element) -> Optional[Reference]:
        """Parse a single bibliographic reference with enhanced citation details"""
        # Title
        title_elem = bibl.find('.//tei:title', self.NAMESPACES)
        title = title_elem.text if title_elem is not None else None
        
        # Authors
        authors = []
        for author in bibl.findall('.//tei:author', self.NAMESPACES):
            name_parts = []
            forename = author.find('.//tei:forename', self.NAMESPACES)
            surname = author.find('.//tei:surname', self.NAMESPACES)
            
            if forename is not None and forename.text:
                name_parts.append(forename.text)
            if surname is not None and surname.text:
                name_parts.append(surname.text)
            
            if name_parts:
                authors.append(' '.join(name_parts))
        
        # Year
        year = None
        date_elem = bibl.find('.//tei:date', self.NAMESPACES)
        if date_elem is not None:
            when = date_elem.get('when')
            if when:
                try:
                    year = int(when[:4])
                except:
                    pass
        
        # Venue
        venue = None
        meeting = bibl.find('.//tei:meeting', self.NAMESPACES)
        if meeting is not None:
            venue = meeting.text
        else:
            journal = bibl.find('.//tei:title[@level="j"]', self.NAMESPACES)
            if journal is not None:
                venue = journal.text
        
        # DOI
        doi = None
        idno = bibl.find('.//tei:idno[@type="DOI"]', self.NAMESPACES)
        if idno is not None:
            doi = idno.text
        
        # Raw text
        raw_text = self._extract_text_from_element(bibl)
        
        if not raw_text and not title:
            return None
        
        # Extract citation details from the document
        cited_by_sections = self._find_citation_sections(bibl, title, authors)
        
        return Reference(
            raw_text=raw_text or "",
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            doi=doi,
            cited_by_sections=cited_by_sections
        )
    
    def _find_citation_sections(self, bibl: ET.Element, title: str, authors: List[str]) -> List[str]:
        """Find sections where this reference is cited"""
        try:
            # Since ElementTree doesn't have getparent(), we'll use a simpler approach
            # Look for citations in the current bibliography element and its context
            cited_sections = []
            
            # Generate citation patterns
            citation_patterns = self._generate_citation_patterns(title, authors)
            
            # Search for citations in the current element and its siblings
            # This is a simplified approach that avoids the getparent() issue
            current_element = bibl
            
            # Try to find the document root by looking for common parent elements
            # Look for body, div, or other structural elements
            for elem in current_element.iter():
                if elem.tag.endswith('body') or elem.tag.endswith('div'):
                    # Check if this element contains citation patterns
                    elem_text = self._extract_text_from_element(elem)
                    if elem_text:
                        for pattern in citation_patterns:
                            if re.search(pattern, elem_text, re.IGNORECASE):
                                # Try to get section title
                                head = elem.find('.//tei:head', self.NAMESPACES)
                                if head is not None and head.text:
                                    section_title = head.text.strip()
                                    if section_title and section_title not in cited_sections:
                                        cited_sections.append(section_title)
                                break
            
            return cited_sections
            
        except Exception as e:
            logger.warning(f"Failed to find citation sections: {e}")
            return []
    
    def _generate_citation_patterns(self, title: str, authors: List[str]) -> List[str]:
        """Generate patterns to search for citations of this reference"""
        patterns = []
        
        if title:
            # Extract key words from title (first 3-5 words)
            title_words = title.split()[:5]
            if len(title_words) >= 3:
                title_pattern = r'\b' + r'\s+'.join(title_words) + r'\b'
                patterns.append(title_pattern)
        
        if authors:
            # Use first author's last name
            first_author = authors[0]
            last_name = first_author.split()[-1] if first_author else ""
            if last_name:
                # Look for author citations like "Smith et al." or "Smith (2020)"
                author_patterns = [
                    rf'\b{last_name}\s+et\s+al\.',
                    rf'\b{last_name}\s*\(\d{{4}}\)',
                    rf'\b{last_name}\s+and\s+',
                    rf'\b{last_name}\s*,\s*\d{{4}}'
                ]
                patterns.extend(author_patterns)
        
        return patterns


    # Enhanced methods for better section detection and page number detection
    
    def _determine_section_type_enhanced(self, title: str) -> SectionType:
        """Enhanced section type detection with more patterns"""
        title_lower = title.lower().strip()
        
        # Check for exact matches first
        for keyword, stype in self.SECTION_MAPPING.items(): # Changed from ENHANCED_SECTION_MAPPING
            if keyword in title_lower:
                return stype
        
        # Check for partial matches
        title_words = set(title_lower.split())
        for keyword, stype in self.SECTION_MAPPING.items(): # Changed from ENHANCED_SECTION_MAPPING
            keyword_words = set(keyword.split())
            if keyword_words.issubset(title_words):
                return stype
        
        return SectionType.OTHER
    
    def _detect_section_level(self, div: ET.Element, title: str) -> int:
        """Detect section level based on structure and title patterns"""
        # Check for numbering patterns in title
        if re.match(r'^\d+\.', title):
            return 1
        elif re.match(r'^\d+\.\d+', title):
            return 2
        elif re.match(r'^\d+\.\d+\.\d+', title):
            return 3
        
        # Check for heading styles (if available in TEI)
        head = div.find('tei:head', self.NAMESPACES)
        if head is not None:
            # Check for style attributes
            style = head.get('style')
            if style:
                if 'h1' in style or 'title' in style:
                    return 1
                elif 'h2' in style or 'subtitle' in style:
                    return 2
                elif 'h3' in style:
                    return 3
        
        # Check for section type to determine level
        # Abstract, Introduction, Conclusion are typically level 1
        title_lower = title.lower()
        if any(keyword in title_lower for keyword in ['abstract', 'introduction', 'conclusion', 'summary']):
            return 1
        
        # Related work, methodology, results, discussion are typically level 1
        if any(keyword in title_lower for keyword in ['related', 'method', 'result', 'discussion']):
            return 1
        
        # Check for subsection indicators in title
        if any(keyword in title_lower for keyword in ['sub', 'subsection', 'part']):
            return 2
        
        # Default to level 1
        return 1
    
    def _get_accurate_page_number_enhanced(self, p_elem: ET.Element, text: str) -> int:
        """Enhanced page number detection using multiple methods"""
        # Method 1: GROBID coordinates (most accurate)
        coords = p_elem.get('coords')
        if coords:
            try:
                # Parse coords format: "page,x1,y1,x2,y2;..."
                page = int(coords.split(',')[0])
                if page > 0:
                    return page
            except:
                pass
        
        # Method 2: Look for page break markers in TEI
        page_break = p_elem.find('.//tei:pb', self.NAMESPACES)
        if page_break is not None:
            page_num = page_break.get('n')
            if page_num:
                try:
                    return int(page_num)
                except:
                    pass
        
        # Method 3: Use PyMuPDF for text-based page detection
        if hasattr(self, 'pdf_path') and self.pdf_path.exists():
            try:
                import fitz
                doc = fitz.open(str(self.pdf_path))
                
                # Search for the text in each page with improved matching
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    page_text = page.get_text()
                    
                    if self._text_matches_page_enhanced(text, page_text):
                        doc.close()
                        return page_num + 1
                
                doc.close()
            except Exception as e:
                logger.warning(f"Failed to use PyMuPDF for page detection: {e}")
        
        # Method 4: Fallback to page 1
        return 1
    
    def _calculate_consecutive_word_score(self, search_text: str, page_text: str) -> float:
        """Calculate score based on consecutive word sequences"""
        search_words = search_text.split()
        page_words = page_text.split()
        
        if len(search_words) < 2:
            return 1.0
        
        consecutive_matches = 0
        total_sequences = len(search_words) - 1
        
        for i in range(len(search_words) - 1):
            sequence = f"{search_words[i]} {search_words[i+1]}"
            if sequence in page_text:
                consecutive_matches += 1
        
        if total_sequences == 0:
            return 1.0
        
        return consecutive_matches / total_sequences
    
    def _text_matches_page_enhanced(self, search_text: str, page_text: str) -> bool:
        """Enhanced text matching for page detection with stricter criteria"""
        # Clean and normalize text
        search_clean = self._normalize_text_enhanced(search_text)
        page_clean = self._normalize_text_enhanced(page_text)
        
        # If search text is very short, require exact match
        if len(search_clean) < 30:
            return search_clean in page_clean
        
        # For longer text, use improved partial matching with stricter criteria
        search_words = set(search_clean.split())
        page_words = set(page_clean.split())
        
        if not search_words:
            return False
        
        # Calculate word overlap with better algorithm
        common_words = search_words.intersection(page_words)
        overlap_ratio = len(common_words) / len(search_words)
        
        # Require at least 75% word overlap for a match (increased threshold)
        if overlap_ratio < 0.75:
            return False
        
        # Additional check: require consecutive word sequences for better quality
        consecutive_score = self._calculate_consecutive_word_score(search_clean, page_clean)
        
        # Final decision: require both high overlap AND good consecutive matching
        return consecutive_score >= 0.6
    
    def _normalize_text_enhanced(self, text: str) -> str:
        """Enhanced text normalization for better matching"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Convert to lowercase for comparison
        text = text.lower()
        # Remove punctuation but keep important separators
        text = re.sub(r'[^\w\s\-]', '', text)
        return text
    
    def _extract_text_from_element_enhanced(self, elem: ET.Element) -> str:
        """Enhanced text extraction with better formatting preservation"""
        texts = []
        
        if elem.text:
            texts.append(elem.text)
        
        for child in elem:
            if child.tag.endswith('ref'):
                # Handle references with better formatting
                if child.text:
                    texts.append(child.text)
            elif child.tag.endswith('hi'):
                # Handle highlighted text (bold, italic, etc.)
                child_text = self._extract_text_from_element_enhanced(child)
                if child_text:
                    texts.append(child_text)
            else:
                # Recursively extract text
                child_text = self._extract_text_from_element_enhanced(child)
                if child_text:
                    texts.append(child_text)
            
            if child.tail:
                texts.append(child.tail)
        
        return ' '.join(texts).strip()
    
    def _extract_paragraphs_enhanced(self, div: ET.Element) -> Tuple[List[Paragraph], set]:
        """Enhanced paragraph extraction with better filtering and detection"""
        paragraphs = []
        page_nums = set()
        
        for p_elem in div.findall('tei:p', self.NAMESPACES):
            text = self._extract_text_from_element_enhanced(p_elem)
            if text and len(text.strip()) > 10:  # Filter out very short paragraphs
                # Use enhanced page detection
                page = self._get_accurate_page_number_enhanced(p_elem, text)
                page_nums.add(page)
                
                # Extract bounding box if available
                bbox = self._extract_bounding_box(p_elem)
                
                # Detect paragraph style
                style = self._detect_paragraph_style(p_elem, text)
                
                paragraphs.append(Paragraph(
                    text=text,
                    page=page,
                    bbox=bbox,
                    style=style
                ))
        
        return paragraphs, page_nums
    
    def _extract_bounding_box(self, elem: ET.Element) -> Optional[BoundingBox]:
        """Extract bounding box coordinates from TEI element"""
        coords = elem.get('coords')
        if not coords:
            return None
        
        try:
            # Parse coords format: "page,x1,y1,x2,y2;..."
            parts = coords.split(',')
            if len(parts) >= 5:
                page = int(parts[0])
                x1 = float(parts[1])
                y1 = float(parts[2])
                x2 = float(parts[3])
                y2 = float(parts[4])
                
                return BoundingBox(
                    page=page,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2
                )
        except:
            pass
        
        return None
    
    def _detect_paragraph_style(self, elem: ET.Element, text: str) -> Optional[Dict[str, Any]]:
        """Detect paragraph style (quote, code, list item, etc.)"""
        style_info = {}
        
        # Check for quote indicators
        if text.startswith('"') and text.endswith('"'):
            style_info['type'] = 'quote'
            style_info['quote_type'] = 'quoted_text'
        
        # Check for code indicators
        elif '`' in text or '```' in text:
            style_info['type'] = 'code'
            style_info['code_type'] = 'inline_code' if '`' in text else 'code_block'
        
        # Check for list indicators
        elif re.match(r'^[\-\*•]\s+', text):
            style_info['type'] = 'list'
            style_info['list_type'] = 'unordered'
            style_info['list_marker'] = text[0]
        elif re.match(r'^\d+\.\s+', text):
            style_info['type'] = 'list'
            style_info['list_type'] = 'ordered'
            style_info['list_marker'] = 'numbered'
        
        # Check for figure/table references
        elif re.search(r'Figure\s+\d+', text, re.IGNORECASE):
            style_info['type'] = 'reference'
            style_info['reference_type'] = 'figure'
            style_info['reference_target'] = 'figure'
        elif re.search(r'Table\s+\d+', text, re.IGNORECASE):
            style_info['type'] = 'reference'
            style_info['reference_type'] = 'table'
            style_info['reference_target'] = 'table'
        
        # Check for mathematical content
        elif re.search(r'[=+\-*/^()\[\]{}]', text):
            style_info['type'] = 'mathematical'
            style_info['math_content'] = True
        
        # Check for citation patterns
        elif re.search(r'\[\d+\]|\(\d+\)|et al\.', text):
            style_info['type'] = 'citation'
            style_info['citation_pattern'] = 'academic'
        
        # Return style info if any was detected, otherwise None
        return style_info if style_info else None
    
    def _extract_section_label(self, div: ET.Element) -> Optional[str]:
        """Extract section numbering/label from TEI element"""
        head = div.find('tei:head', self.NAMESPACES)
        if head is None:
            return None
        
        # Check for label attribute
        label = head.get('n')
        if label:
            return label
        
        # Check for numbering in title
        title = head.text or ""
        if re.match(r'^\d+\.', title):
            return title.split('.')[0] + '.'
        
        return None
    
    def _calculate_section_page_range(self, paragraphs: List[Paragraph]) -> Tuple[int, int]:
        """Calculate accurate page start and end for a section"""
        if not paragraphs:
            return 1, 1
        
        pages = [p.page for p in paragraphs if p.page > 0]
        if not pages:
            return 1, 1
        
        return min(pages), max(pages)
    
    def _extract_subsections(self, div: ET.Element, parent_level: int) -> List[Section]:
        """Extract nested subsections from a div element"""
        subsections = []
        subsection_id = 0
        
        for child_div in div.findall('tei:div', self.NAMESPACES):
            subsection = self._parse_section_enhanced(child_div, subsection_id)
            if subsection:
                subsection.level = parent_level + 1
                subsections.append(subsection)
                subsection_id += 1
        
        return subsections
    
    def _calculate_accurate_section_page_range(self, page_nums: set, paragraphs: List[Paragraph]) -> Tuple[int, int]:
        """Calculate accurate page start and end for a section with validation"""
        if not page_nums:
            return 1, 1
        
        # Filter out invalid page numbers
        valid_pages = [p for p in page_nums if p > 0]
        
        if not valid_pages:
            return 1, 1
        
        # Get page numbers from paragraphs as backup
        para_pages = [p.page for p in paragraphs if hasattr(p, 'page') and p.page > 0]
        
        # Combine both sources and remove duplicates
        all_pages = list(set(valid_pages + para_pages))
        
        if not all_pages:
            return 1, 1
        
        # Sort pages and find range
        all_pages.sort()
        page_start = all_pages[0]
        page_end = all_pages[-1]
        
        # Validate page range
        if page_end < page_start:
            page_end = page_start
        
        # If section spans too many pages, it might be incorrect - limit it
        max_reasonable_span = 10  # Most sections don't span more than 10 pages
        if page_end - page_start > max_reasonable_span:
            # Look for natural breaks in the middle
            page_mid = (page_start + page_end) // 2
            # Check if there's a concentration of content around the middle
            mid_pages = [p for p in all_pages if abs(p - page_mid) <= 2]
            if mid_pages:
                page_start = min(mid_pages)
                page_end = max(mid_pages)
        
        return page_start, page_end


# Global instance
grobid_extractor = GROBIDExtractor()