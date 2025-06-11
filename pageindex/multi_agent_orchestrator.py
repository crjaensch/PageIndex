import asyncio
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from agents import Agent

# Import existing agents and utilities
from pageindex.page_index import (
    CHECK_TITLE_APPEARANCE_AGENT, CHECK_TITLE_START_AGENT, TOC_DETECTOR_AGENT,
    TOC_EXTRACTION_COMPLETE_AGENT, EXTRACT_TOC_AGENT, TOC_JSON_TRANSFORMER_AGENT,
    CREATE_TOC_FROM_CONTENT_AGENT, SINGLE_TOC_ITEM_FIXER_AGENT, 
    TOC_TRANSFORMATION_COMPLETE_AGENT, ADD_PAGE_NUMBER_AGENT,
    PAGE_INDEX_DETECTOR_AGENT, page_list_to_group_text
)
from pageindex.utils import (run_specific_agent, extract_json, JsonLogger, get_pdf_name, 
                  get_page_tokens, ConfigLoader, DEFAULT_AGENT_MODEL, count_tokens,
                  convert_physical_index_to_int, post_processing, 
                  NODE_SUMMARY_AGENT, DOC_DESCRIPTION_AGENT)


# Agent wrapper for consistent interface
@dataclass
class AgentTask:
    """Represents a task for an agent to complete"""
    agent_name: str
    input_data: Dict[str, Any]
    expected_output_type: str
    dependencies: List[str] = None  # List of agent names that must complete first
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass 
class AgentResult:
    """Standardized agent result format"""
    agent_name: str
    success: bool
    data: Any
    confidence: float
    metadata: Dict[str, Any]
    execution_time: float
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class EnhancedPageIndexProcessingContext:
    """Enhanced context that provides just-in-time data preparation for agents"""
    
    def __init__(self, doc, opt=None, logger=None, model=None):
        # Core document data
        self.doc = doc
        self.doc_name = get_pdf_name(doc)
        self.page_list: Optional[List] = None
        self.opt = opt if opt else ConfigLoader().load({})
        self.model = model if model else getattr(self.opt, 'model', DEFAULT_AGENT_MODEL)
        self.logger = logger if logger else JsonLogger(doc)
        
        # Document characteristics (populated by analysis)
        self.total_pages = len(doc.pages) if hasattr(doc, 'pages') else 0
        self.total_tokens = 0
        self.document_type = "unknown"
        self.complexity_level = "unknown"
        self.estimated_sections = 0
        
        # TOC-related data (populated by detection agents)
        self.has_toc = False
        self.toc_pages: List[int] = []
        self.raw_toc_content = ""
        self.toc_has_page_numbers = False
        self.toc_extraction_complete = False
        self.toc_transformation_complete = False
        self.structured_toc: List[Dict] = []
        self.start_page_index = 1  # Default start page index
        
        # Processing results (populated by processing agents)
        self.toc_with_physical_indices: List[Dict] = []
        self.verified_toc: List[Dict] = []
        self.final_structure: List[Dict] = []
        self.incorrect_items: List[Dict] = []
        self.page_offset = 0  # For mapping between TOC page numbers and physical indices
        
        # Enhancement data (populated by enhancement agents)
        self.node_summaries: Dict[str, str] = {}
        self.document_description = ""
        
        # Processing strategy
        self.processing_strategy = "unknown"  # Options: "toc_with_page_numbers", "toc_no_page_numbers", "no_toc"
        
        # Cached data for just-in-time preparation
        self._grouped_content_cache: Optional[List[str]] = None
        self._labeled_pages_cache: Optional[str] = None
        self._verification_data_cache: Optional[Dict] = None
        self._labeled_cache: Dict[str, str] = {}
        
        # Agent communication data
        self.agent_results: Dict[str, Any] = {}
        self.processing_metadata = {
            "agents_used": [],
            "strategy_adaptations": [],
            "processing_time": {},
            "execution_timeline": [],
            "toc_detection": {},
            "toc_processing": {},
            "verification": {},
            "correction": {}
        }
    
    def get_sample_pages_for_analysis(self, num_pages=5):
        """Get sample pages for document analysis"""
        # Get pages from beginning, middle, and end
        sample_indices = [0]  # Always include first page
        
        if self.total_pages > 1:
            sample_indices.append(self.total_pages - 1)  # Last page
        
        # Add middle pages if available
        remaining = num_pages - len(sample_indices)
        if remaining > 0 and self.total_pages > 2:
            step = max(1, (self.total_pages - 2) // (remaining + 1))
            middle_indices = list(range(1, self.total_pages - 1, step))[:remaining]
            sample_indices.extend(middle_indices)
        
        # Sort indices
        sample_indices.sort()
        
        # Get page content
        sample_pages = []
        for idx in sample_indices:
            page_text = self.doc.pages[idx].extract_text()
            sample_pages.append({
                "page_number": idx + 1,  # 1-indexed for display
                "content": page_text
            })
        
        return sample_pages
    
    def update_from_document_analysis(self, analysis_data):
        """Update context with document analysis results"""
        if not isinstance(analysis_data, dict):
            return
            
        # Update document characteristics
        self.document_type = analysis_data.get("document_type", self.document_type)
        self.complexity_level = analysis_data.get("complexity_level", self.complexity_level)
        self.language = analysis_data.get("language", self.language)
        
        # Log the update
        self.logger.info({
            'event': 'document_analysis_updated',
            'document_type': self.document_type,
            'complexity_level': self.complexity_level,
            'language': self.language
        })
    
    def get_content_for_toc_extraction(self, page_indices):
        """Get content from specified pages for TOC extraction"""
        content = []
        
        for idx in page_indices:
            if 0 <= idx < self.total_pages:
                page_text = self.doc.pages[idx].extract_text()
                content.append({
                    "page_number": idx + 1,  # 1-indexed for display
                    "content": page_text
                })
        
        return content
    
    def get_grouped_content_for_toc_generation(self, chunk_size=5):
        """Group document content for TOC generation when no TOC is found"""
        import math
        
        # Determine number of chunks
        num_chunks = max(1, math.ceil(self.total_pages / chunk_size))
        
        # Group pages into chunks
        content_chunks = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, self.total_pages)
            
            chunk_content = []
            for idx in range(start_idx, end_idx):
                page_text = self.doc.pages[idx].extract_text()
                chunk_content.append({
                    "page_number": idx + 1,  # 1-indexed for display
                    "content": page_text
                })
            
            content_chunks.append(chunk_content)
        
        return content_chunks
    
    def get_section_content_for_summary(self, start_page, end_page, max_pages=5):
        """Get content from a section for summarization"""
        # Limit the number of pages to process
        if end_page - start_page > max_pages:
            # Sample pages from the section
            sample_indices = [start_page]  # Start page
            sample_indices.append(end_page - 1)  # End page
            
            # Add middle pages
            remaining = max_pages - 2
            if remaining > 0:
                step = max(1, (end_page - start_page - 2) // remaining)
                for i in range(remaining):
                    middle_idx = start_page + 1 + i * step
                    if middle_idx < end_page - 1:
                        sample_indices.append(middle_idx)
            
            # Sort indices
            sample_indices.sort()
        else:
            # Use all pages in the section
            sample_indices = list(range(start_page, end_page))
        
        # Get content from selected pages
        section_content = []
        for idx in sample_indices:
            if 0 <= idx < self.total_pages:
                page_text = self.doc.pages[idx].extract_text()
                section_content.append({
                    "page_number": idx + 1,  # 1-indexed for display
                    "content": page_text
                })
        
        return section_content
        
    async def find_toc_pages(self, start_page=0, max_pages=10) -> List[int]:
        """Find pages containing table of contents with detailed logging"""
        self.logger.info({'event': 'toc_detection_started'})
        
        toc_pages = []
        for page_idx in range(start_page, min(start_page + max_pages, self.total_pages)):
            is_toc = await self._detect_toc_in_page(page_idx)
            
            if is_toc:
                toc_pages.append(page_idx)
                self.logger.info({
                    'event': 'toc_page_detected',
                    'page_number': page_idx + 1,  # 1-indexed for display
                    'physical_index': page_idx
                })
        
        # Update context state
        self.has_toc = len(toc_pages) > 0
        self.toc_pages = toc_pages
        
        self.logger.info({
            'event': 'toc_detection_complete',
            'has_toc': self.has_toc,
            'toc_pages_count': len(toc_pages),
            'toc_pages': toc_pages
        })
        
        return toc_pages
    
    async def _detect_toc_in_page(self, page_idx) -> bool:
        """Detect if a page contains table of contents"""
        if page_idx < 0 or page_idx >= self.total_pages:
            return False
            
        # Get page content
        page_text = self.doc.pages[page_idx].extract_text()
        
        # Create task for TOC detector agent
        task_data = {
            "page_content": page_text,
            "page_number": page_idx + 1  # 1-indexed for display
        }
        
        # Execute agent
        result = await run_specific_agent(TOC_DETECTOR_AGENT, task_data, self.model)
        
        # Extract result
        json_result = extract_json(result)
        
        # Check if page contains TOC
        is_toc = json_result.get("is_toc", False) if json_result else False
        
        return is_toc
    
    async def _extract_toc_from_page(self, page_idx) -> str:
        """Extract TOC content from a page"""
        if page_idx < 0 or page_idx >= self.total_pages:
            return ""
            
        # Get page content
        page_text = self.doc.pages[page_idx].extract_text()
        
        # Create task for TOC extractor agent
        task_data = {
            "page_content": page_text,
            "page_number": page_idx + 1  # 1-indexed for display
        }
        
        # Execute agent
        result = await run_specific_agent(EXTRACT_TOC_AGENT, task_data, self.model)
        
        # Extract TOC content
        toc_content = result.strip()
        
        return toc_content
    
    async def extract_toc_from_pages(self) -> str:
        """Extract TOC content from identified TOC pages"""
        if not self.toc_pages:
            self.logger.warning({'event': 'toc_extraction_failed', 'reason': 'no_toc_pages'})
            return ""
        
        self.logger.info({'event': 'toc_extraction_started'})
        
        # Extract TOC from each page
        toc_content = ""
        for page_idx in self.toc_pages:
            page_toc = await self._extract_toc_from_page(page_idx)
            if page_toc:
                if toc_content:
                    toc_content += "\n\n"
                toc_content += page_toc
                
                self.logger.info({
                    'event': 'toc_page_extracted',
                    'page_number': page_idx + 1,  # 1-indexed for display
                    'content_length': len(page_toc)
                })
        
        # Update context state
        self.raw_toc_content = toc_content
        self.toc_extraction_complete = await self._check_toc_extraction_complete(toc_content)
        
        self.logger.info({
            'event': 'toc_extraction_complete',
            'content_length': len(toc_content),
            'extraction_complete': self.toc_extraction_complete
        })
        
        return toc_content
    
    async def _check_toc_extraction_complete(self, toc_content, extracted_toc=None) -> bool:
        """Check if TOC extraction is complete"""
        if not toc_content:
            return False
        
        # Create task for TOC extraction completeness checker
        task_data = {
            "toc_content": toc_content,
            "extracted_toc": extracted_toc or self.raw_toc_content
        }
        
        # Execute agent
        result = await run_specific_agent(TOC_EXTRACTION_COMPLETE_AGENT, task_data, self.model)
        
        # Extract result
        json_result = extract_json(result)
        
        # Check if extraction is complete
        is_complete = json_result.get("is_complete", False) if json_result else False
        
        return is_complete
    
    async def detect_page_numbers_in_toc(self) -> bool:
        """Detect if TOC contains page numbers"""
        if not self.raw_toc_content:
            return False
        
        self.logger.info({'event': 'page_number_detection_started'})
        
        # Create task for page index detector agent
        task_data = {
            "toc_content": self.raw_toc_content
        }
        
        # Execute agent
        result = await run_specific_agent(PAGE_INDEX_DETECTOR_AGENT, task_data, self.model)
        
        # Extract result
        json_result = extract_json(result)
        
        # Check if TOC has page numbers
        has_page_numbers = json_result.get("has_page_numbers", False) if json_result else False
        
        # Update context state
        self.toc_has_page_numbers = has_page_numbers
        
        # If page numbers are detected, also get the start page index
        if has_page_numbers and json_result and "start_page_index" in json_result:
            self.start_page_index = json_result["start_page_index"]
        
        self.logger.info({
            'event': 'page_number_detection_complete',
            'has_page_numbers': has_page_numbers,
            'start_page_index': self.start_page_index
        })
        
        return has_page_numbers
    
    async def transform_toc_to_json(self) -> List[Dict]:
        """Transform raw TOC content to structured JSON format"""
        if not self.raw_toc_content:
            self.logger.warning({'event': 'toc_transformation_failed', 'reason': 'no_raw_content'})
            return []
        
        self.logger.info({'event': 'toc_transformation_started'})
        
        # Create task for TOC transformer agent
        task_data = {
            "toc_content": self.raw_toc_content,
            "has_page_numbers": self.toc_has_page_numbers
        }
        
        # Execute agent
        result = await run_specific_agent(TOC_JSON_TRANSFORMER_AGENT, task_data, self.model)
        
        # Extract result
        json_result = extract_json(result)
        
        # Get structured TOC
        structured_toc = json_result.get("table_of_contents", []) if json_result else []
        
        # Update context state
        self.structured_toc = structured_toc
        self.toc_transformation_complete = await self._check_toc_transformation_complete()
        
        self.logger.info({
            'event': 'toc_transformation_complete',
            'items_count': len(structured_toc),
            'transformation_complete': self.toc_transformation_complete
        })
        
        return structured_toc
    
    async def _check_toc_transformation_complete(self) -> bool:
        """Check if TOC transformation is complete and accurate"""
        if not self.structured_toc:
            return False
        
        # Create task for TOC transformation completeness checker
        task_data = {
            "raw_toc": self.raw_toc_content,
            "structured_toc": self.structured_toc
        }
        
        # Execute agent
        result = await run_specific_agent(TOC_TRANSFORMATION_COMPLETE_AGENT, task_data, self.model)
        
        # Extract result
        json_result = extract_json(result)
        
        # Check if transformation is complete
        is_complete = json_result.get("is_complete", False) if json_result else False
        
        return is_complete
    
    async def add_page_numbers_to_toc(self) -> List[Dict]:
        """Add physical page numbers to TOC items"""
        if not self.structured_toc:
            self.logger.warning({'event': 'page_number_mapping_failed', 'reason': 'no_structured_toc'})
            return []
        
        self.logger.info({'event': 'page_number_mapping_started'})
        
        # Create task for page number mapper agent
        task_data = {
            "structured_toc": self.structured_toc,
            "has_page_numbers": self.toc_has_page_numbers,
            "total_pages": self.total_pages,
            "start_page_index": self.start_page_index
        }
        
        # Execute agent
        result = await run_specific_agent(ADD_PAGE_NUMBER_AGENT, task_data, self.model)
        
        # Extract result
        json_result = extract_json(result)
        
        # Get TOC with physical indices
        toc_with_indices = json_result.get("toc_with_physical_indices", []) if json_result else []
        
        # Update context state
        self.toc_with_physical_indices = toc_with_indices
        
        # Get page offset if available
        if json_result and "page_offset" in json_result:
            self.page_offset = json_result["page_offset"]
        
        self.logger.info({
            'event': 'page_number_mapping_complete',
            'items_with_physical_index': sum(1 for item in toc_with_indices if item.get('physical_index') is not None),
            'page_offset': self.page_offset
        })
        
        return toc_with_indices
    
    async def validate_and_truncate_physical_indices(self) -> List[Dict]:
        """Validate and truncate physical indices that exceed document length"""
        if not self.toc_with_physical_indices:
            return []
        
        # Validate and truncate physical indices
        for item in self.toc_with_physical_indices:
            if 'physical_index' in item:
                # Ensure physical index is within document bounds
                item['physical_index'] = max(0, min(item['physical_index'], self.total_pages - 1))
        
        self.logger.info({'event': 'physical_indices_validated'})
        
        return self.toc_with_physical_indices
    
    async def verify_toc(self) -> List[Dict]:
        """Verify TOC entries against document content"""
        if not self.toc_with_physical_indices:
            self.logger.warning({'event': 'toc_verification_failed', 'reason': 'no_toc_with_indices'})
            return []
        
        self.logger.info({'event': 'toc_verification_started'})
        
        # Prepare verification data
        verification_data = await self._prepare_verification_data()
        
        # Create task for TOC verifier agent
        task_data = {
            "toc_with_physical_indices": self.toc_with_physical_indices,
            "verification_data": verification_data,
            "total_pages": self.total_pages
        }
        
        # Execute agent
        result = await run_specific_agent(VERIFY_TOC_AGENT, task_data, self.model)
        
        # Extract result
        json_result = extract_json(result)
        
        if not json_result:
            self.logger.warning({'event': 'toc_verification_failed', 'reason': 'invalid_result'})
            return self.toc_with_physical_indices
        
        # Get verified TOC and incorrect items
        verified_toc = json_result.get("verified_toc", [])
        incorrect_items = json_result.get("incorrect_items", [])
        
        # Update context state
        self.verified_toc = verified_toc
        self.incorrect_items = incorrect_items
        
        self.logger.info({
            'event': 'toc_verification_complete',
            'verified_items': len(verified_toc),
            'incorrect_items': len(incorrect_items)
        })
        
        return verified_toc
    
    async def _prepare_verification_data(self) -> Dict:
        """Prepare data for TOC verification"""
        verification_data = {}
        
        # For each TOC item with a physical index, get the page content
        for item in self.toc_with_physical_indices:
            physical_index = item.get('physical_index')
            if physical_index is not None and 0 <= physical_index < self.total_pages:
                # Get content from the page and surrounding pages
                start_idx = max(0, physical_index - 1)
                end_idx = min(self.total_pages, physical_index + 2)
                
                page_contents = []
                for idx in range(start_idx, end_idx):
                    page_text = self.doc.pages[idx].extract_text()
                    page_contents.append({
                        "page_number": idx + 1,  # 1-indexed for display
                        "content": page_text
                    })
                
                # Store verification data for this item
                item_key = f"{item.get('title')}_{physical_index}"
                verification_data[item_key] = {
                    "title": item.get('title'),
                    "physical_index": physical_index,
                    "page_contents": page_contents
                }
        
        return verification_data
    
    async def fix_incorrect_toc(self) -> List[Dict]:
        """Fix incorrect TOC entries"""
        if not self.incorrect_items:
            return self.verified_toc or self.toc_with_physical_indices
        
        self.logger.info({'event': 'toc_correction_started'})
        
        # Prepare correction data
        correction_data = await self._prepare_correction_data()
        
        # Create task for TOC corrector agent
        task_data = {
            "incorrect_items": self.incorrect_items,
            "correction_data": correction_data,
            "total_pages": self.total_pages
        }
        
        # Execute agent
        result = await run_specific_agent(FIX_TOC_AGENT, task_data, self.model)
        
        # Extract result
        json_result = extract_json(result)
        
        if not json_result:
            self.logger.warning({'event': 'toc_correction_failed', 'reason': 'invalid_result'})
            return self.verified_toc or self.toc_with_physical_indices
        
        # Get corrected items
        corrected_items = json_result.get("corrected_items", [])
        
        # Update verified TOC with corrected items
        if self.verified_toc:
            # Replace incorrect items with corrected ones
            corrected_toc = []
            for item in self.verified_toc:
                if item.get("verification_status") == "incorrect":
                    # Find corrected version
                    corrected_version = next(
                        (c for c in corrected_items if c.get("title") == item.get("title")),
                        item
                    )
                    corrected_toc.append(corrected_version)
                else:
                    corrected_toc.append(item)
            
            self.verified_toc = corrected_toc
        else:
            # Create verified TOC from toc_with_physical_indices and corrected items
            corrected_toc = []
            for item in self.toc_with_physical_indices:
                if any(i.get("title") == item.get("title") for i in self.incorrect_items):
                    # Find corrected version
                    corrected_version = next(
                        (c for c in corrected_items if c.get("title") == item.get("title")),
                        item
                    )
                    corrected_toc.append(corrected_version)
                else:
                    item_with_status = item.copy()
                    item_with_status["verification_status"] = "correct"
                    corrected_toc.append(item_with_status)
            
            self.verified_toc = corrected_toc
        
        # Update incorrect items list
        self.incorrect_items = [item for item in self.verified_toc if item.get("verification_status") == "incorrect"]
        
        self.logger.info({
            'event': 'toc_correction_complete',
            'corrected_items': len(corrected_items),
            'remaining_incorrect': len(self.incorrect_items)
        })
        
        return self.verified_toc
    
    async def _prepare_correction_data(self) -> Dict:
        """Prepare data for TOC correction"""
        correction_data = {}
        
        # For each incorrect item, get content from surrounding pages
        for item in self.incorrect_items:
            title = item.get('title')
            physical_index = item.get('physical_index')
            
            if physical_index is not None:
                # Get content from a wider range of pages
                start_idx = max(0, physical_index - 3)
                end_idx = min(self.total_pages, physical_index + 4)
                
                page_contents = []
                for idx in range(start_idx, end_idx):
                    page_text = self.doc.pages[idx].extract_text()
                    page_contents.append({
                        "page_number": idx + 1,  # 1-indexed for display
                        "physical_index": idx,
                        "content": page_text
                    })
                
                # Store correction data for this item
                item_key = f"{title}_{physical_index}"
                correction_data[item_key] = {
                    "title": title,
                    "current_physical_index": physical_index,
                    "page_contents": page_contents
                }
        
        return correction_data
    
    async def initialize(self):
        """Initialize basic document data"""
        start_time = asyncio.get_event_loop().time()
        self.page_list = get_page_tokens(self.doc)
        self.total_pages = len(self.page_list)
        self.total_tokens = sum(page[1] for page in self.page_list)
        
        self.logger.info({
            'event': 'initialization_complete',
            'total_pages': self.total_pages,
            'total_tokens': self.total_tokens,
            'initialization_time': asyncio.get_event_loop().time() - start_time
        })
        return self
        
    async def find_toc_pages(self, start_page_index=0, max_pages=10):
        """Find pages containing table of contents"""
        self.logger.info({
            'event': 'toc_detection_started',
            'start_page': start_page_index,
            'max_pages': max_pages
        })
        
        toc_pages = []
        current_toc = ""
        is_complete = False
        
        # Check pages for TOC presence
        for page_idx in range(start_page_index, min(start_page_index + max_pages, self.total_pages)):
            page_content = self.page_list[page_idx][0]
            
            # Check if this page contains TOC
            toc_detected = await self._detect_toc_in_page(page_content)
            
            if toc_detected == 'yes':
                toc_pages.append(page_idx)
                
                # Extract TOC content from this page
                page_toc = await self._extract_toc_from_page(page_content)
                current_toc += page_toc
                
                # Check if TOC extraction is complete
                is_complete = await self._check_toc_extraction_complete(page_content, current_toc)
                
                if is_complete:
                    self.logger.info({
                        'event': 'toc_extraction_complete',
                        'toc_pages': toc_pages
                    })
                    break
        
        # Update context with findings
        self.toc_pages = toc_pages
        self.has_toc = len(toc_pages) > 0
        self.toc_extraction_complete = is_complete
        
        self.processing_metadata["toc_detection"] = {
            "toc_found": self.has_toc,
            "toc_pages": toc_pages,
            "extraction_complete": is_complete
        }
        
        self.logger.info({
            'event': 'toc_detection_complete',
            'has_toc': self.has_toc,
            'toc_pages': toc_pages,
            'toc_extraction_complete': is_complete
        })
        
        return toc_pages
    
    async def _detect_toc_in_page(self, page_content):
        """Detect if a page contains table of contents"""
        user_prompt = f"""Given text:
---
{page_content}
---
Please detect if a table of contents is present and reply in the JSON format as specified in your instructions."""
        
        raw_response = await run_specific_agent(TOC_DETECTOR_AGENT, user_prompt)
        json_content = extract_json(raw_response)
        
        toc_detected = 'no'  # Default
        if isinstance(json_content, dict) and 'toc_detected' in json_content:
            toc_detected = json_content['toc_detected']
            
        return toc_detected
    
    async def _extract_toc_from_page(self, page_content):
        """Extract TOC content from a page"""
        user_prompt = f"""Please extract the full table of contents from the following text:
---
{page_content}
---
Remember to replace '...' with ':' where appropriate and return only the complete table of contents text."""
        
        toc_text = await run_specific_agent(EXTRACT_TOC_AGENT, user_prompt)
        
        if "Error: Agent execution failed" in str(toc_text):
            self.logger.error(f"TOC extraction failed: {toc_text}")
            return ""
            
        return toc_text
    
    async def _check_toc_extraction_complete(self, content, toc):
        """Check if TOC extraction is complete"""
        user_prompt = f"""Given text:
---
{content}
---
Given table of contents (potentially partial or continued):
---
{toc}
---
Is the table of contents extraction complete? Please reply in the JSON format specified in your instructions."""
        
        raw_response = await run_specific_agent(TOC_EXTRACTION_COMPLETE_AGENT, user_prompt)
        json_content = extract_json(raw_response)
        
        is_complete = 'no'  # Default
        if isinstance(json_content, dict) and 'complete' in json_content:
            is_complete = json_content['complete']
            
        return is_complete == 'yes'
    
    async def detect_page_numbers_in_toc(self):
        """Detect if TOC contains page numbers"""
        if not self.raw_toc_content:
            self.logger.warning("Cannot detect page numbers: No TOC content available")
            return False
            
        self.logger.info({
            'event': 'page_number_detection_started',
            'toc_content_length': len(self.raw_toc_content)
        })
        
        user_prompt = f"""Given table of contents text:
---
{self.raw_toc_content}
---
Are page numbers/indices present in this table of contents?
Please reply in the JSON format specified in your instructions for PAGE_INDEX_DETECTOR_AGENT."""
        
        raw_response = await run_specific_agent(PAGE_INDEX_DETECTOR_AGENT, user_prompt)
        json_content = extract_json(raw_response)
        
        page_index_given = 'no'  # Default
        if isinstance(json_content, dict) and 'page_index_given_in_toc' in json_content:
            page_index_given = json_content['page_index_given_in_toc']
            
        has_page_numbers = page_index_given == 'yes'
        self.toc_has_page_numbers = has_page_numbers
        
        self.processing_metadata["toc_processing"]["has_page_numbers"] = has_page_numbers
        
        self.logger.info({
            'event': 'page_number_detection_complete',
            'has_page_numbers': has_page_numbers
        })
        
        return has_page_numbers
        
    async def transform_toc_to_json(self):
        """Transform raw TOC content to structured JSON format"""
        if not self.raw_toc_content:
            self.logger.warning("Cannot transform TOC: No TOC content available")
            return []
            
        self.logger.info({
            'event': 'toc_transformation_started',
            'toc_content_length': len(self.raw_toc_content)
        })
        
        user_prompt = f"""Transform this raw TOC to JSON:
---
{self.raw_toc_content}
---"""
        
        raw_response = await run_specific_agent(TOC_JSON_TRANSFORMER_AGENT, user_prompt)
        json_content = extract_json(raw_response)
        
        structured_toc = []
        if isinstance(json_content, dict) and "table_of_contents" in json_content:
            structured_toc = json_content["table_of_contents"]
        
        # Add list indices for easier tracking
        for i, item in enumerate(structured_toc):
            item['list_index'] = i
        
        self.structured_toc = structured_toc
        
        # Check if transformation is complete
        is_complete = await self._check_toc_transformation_complete()
        self.toc_transformation_complete = is_complete
        
        self.processing_metadata["toc_processing"].update({
            "structured_items": len(structured_toc),
            "transformation_complete": is_complete
        })
        
        self.logger.info({
            'event': 'toc_transformation_complete',
            'structured_items': len(structured_toc),
            'transformation_complete': is_complete
        })
        
        return structured_toc
    
    async def _check_toc_transformation_complete(self):
        """Check if TOC transformation is complete and accurate"""
        if not self.raw_toc_content or not self.structured_toc:
            return False
            
        user_prompt = f"""Raw Table of contents:
---
{self.raw_toc_content}
---
Cleaned/Structured Table of contents:
---
{json.dumps(self.structured_toc, indent=2)}
---
Based on these, is the cleaned/structured ToC complete and accurate representation of the raw ToC?
Please reply in the JSON format specified in your instructions for TOC_TRANSFORMATION_COMPLETE_AGENT."""
        
        raw_response = await run_specific_agent(TOC_TRANSFORMATION_COMPLETE_AGENT, user_prompt)
        json_content = extract_json(raw_response)
        
        completed_status = 'no'  # Default
        if isinstance(json_content, dict) and 'completed' in json_content:
            completed_status = json_content['completed']
            
        return completed_status == 'yes'
    
    async def add_page_numbers_to_toc(self):
        """Add physical page numbers to TOC items"""
        if not self.structured_toc:
            self.logger.warning("Cannot add page numbers: No structured TOC available")
            return []
            
        self.logger.info({
            'event': 'page_number_mapping_started',
            'structured_items': len(self.structured_toc)
        })
        
        # Prepare content with physical indices for mapping
        labeled_content = self.get_labeled_pages_for_mapping(1, self.total_pages)
        
        user_prompt = f"""Partial Document Text:
---
{labeled_content}
---
Given ToC Structure:
---
{json.dumps(self.structured_toc, indent=2)}
---
Map titles to physical page numbers."""
        
        raw_response = await run_specific_agent(ADD_PAGE_NUMBER_AGENT, user_prompt)
        mapped_toc = extract_json(raw_response)
        
        if not isinstance(mapped_toc, list):
            self.logger.error("Failed to map page numbers: Invalid response format")
            mapped_toc = []
        
        # Convert physical_index strings to integers
        for item in mapped_toc:
            if item.get('physical_index'):
                # Extract number from format like "<physical_index_123>"
                match = re.search(r'<physical_index_(\d+)>', item['physical_index'])
                if match:
                    item['physical_index'] = int(match.group(1))
                else:
                    item['physical_index'] = None
        
        self.toc_with_physical_indices = mapped_toc
        
        self.processing_metadata["toc_processing"].update({
            "mapped_items": len(mapped_toc),
            "items_with_physical_index": sum(1 for item in mapped_toc if item.get('physical_index') is not None)
        })
        
        self.logger.info({
            'event': 'page_number_mapping_complete',
            'mapped_items': len(mapped_toc),
            'items_with_physical_index': sum(1 for item in mapped_toc if item.get('physical_index') is not None)
        })
        
        return mapped_toc
        
    async def validate_and_truncate_physical_indices(self):
        """Validate and truncate physical indices that exceed document length"""
        if not self.toc_with_physical_indices:
            return []
            
        self.logger.info({
            'event': 'physical_index_validation_started',
            'total_pages': self.total_pages
        })
        
        valid_items = []
        truncated_items = []
        
        for item in self.toc_with_physical_indices:
            if item.get('physical_index') is not None:
                if item['physical_index'] > self.total_pages:
                    # Truncate to last page
                    truncated_items.append({
                        'title': item['title'],
                        'original_index': item['physical_index'],
                        'truncated_to': self.total_pages
                    })
                    item['physical_index'] = self.total_pages
                    
                # Ensure index is not less than start_page_index
                if item['physical_index'] < self.start_page_index:
                    item['physical_index'] = self.start_page_index
                    
            valid_items.append(item)
        
        self.toc_with_physical_indices = valid_items
        
        if truncated_items:
            self.logger.warning({
                'event': 'physical_indices_truncated',
                'truncated_items': truncated_items
            })
        
        self.logger.info({
            'event': 'physical_index_validation_complete',
            'valid_items': len(valid_items),
            'truncated_items': len(truncated_items)
        })
        
        return valid_items
        
    async def verify_toc(self):
        """Verify TOC entries against document content"""
        if not self.toc_with_physical_indices:
            self.logger.warning("Cannot verify TOC: No TOC with physical indices available")
            return []
            
        self.logger.info({
            'event': 'toc_verification_started',
            'items_to_verify': len(self.toc_with_physical_indices)
        })
        
        # Prepare verification data
        verification_data = self._prepare_verification_data()
        
        user_prompt = f"""Document Content Samples:
---
{verification_data}
---
TOC Structure with Physical Indices:
---
{json.dumps(self.toc_with_physical_indices, indent=2)}
---
Verify if each TOC entry correctly points to its content in the document."""
        
        raw_response = await run_specific_agent(VERIFY_TOC_AGENT, user_prompt)
        verification_result = extract_json(raw_response)
        
        verified_toc = []
        incorrect_items = []
        
        if isinstance(verification_result, dict) and "verified_toc" in verification_result:
            verified_toc = verification_result["verified_toc"]
            
            # Extract incorrect items
            for item in verified_toc:
                if item.get("verification_status") == "incorrect":
                    incorrect_items.append(item)
        
        self.verified_toc = verified_toc
        self.incorrect_items = incorrect_items
        
        self.processing_metadata["verification"] = {
            "total_items": len(verified_toc),
            "correct_items": len(verified_toc) - len(incorrect_items),
            "incorrect_items": len(incorrect_items)
        }
        
        self.logger.info({
            'event': 'toc_verification_complete',
            'total_items': len(verified_toc),
            'correct_items': len(verified_toc) - len(incorrect_items),
            'incorrect_items': len(incorrect_items)
        })
        
        return verified_toc
    
    def _prepare_verification_data(self):
        """Prepare data for TOC verification"""
        if self._verification_data_cache:
            return self._verification_data_cache
            
        verification_data = ""
        
        # For each TOC item with a physical index, get content from that page
        for item in self.toc_with_physical_indices:
            if item.get('physical_index') is not None:
                page_idx = item['physical_index']
                if 0 <= page_idx < self.total_pages:
                    page_content = self.page_list[page_idx][0][:500]  # Get first 500 chars of the page
                    verification_data += f"\n--- Page {page_idx} (TOC entry: {item['title']}) ---\n{page_content}\n"
        
        self._verification_data_cache = verification_data
        return verification_data
    
    async def fix_incorrect_toc(self):
        """Fix incorrect TOC entries"""
        if not self.incorrect_items:
            self.logger.info("No incorrect TOC items to fix")
            return self.verified_toc
            
        self.logger.info({
            'event': 'toc_correction_started',
            'incorrect_items': len(self.incorrect_items)
        })
        
        # Prepare content for correction
        correction_data = self._prepare_correction_data()
        
        user_prompt = f"""Document Content Samples:
---
{correction_data}
---
Incorrect TOC Items:
---
{json.dumps(self.incorrect_items, indent=2)}
---
Full TOC Structure:
---
{json.dumps(self.verified_toc, indent=2)}
---
Fix the incorrect TOC entries by finding the correct physical page indices."""
        
        raw_response = await run_specific_agent(FIX_TOC_AGENT, user_prompt)
        correction_result = extract_json(raw_response)
        
        corrected_toc = self.verified_toc
        
        if isinstance(correction_result, dict) and "corrected_items" in correction_result:
            corrected_items = correction_result["corrected_items"]
            
            # Update the verified TOC with corrected items
            for corrected_item in corrected_items:
                for i, item in enumerate(corrected_toc):
                    if item.get("list_index") == corrected_item.get("list_index"):
                        corrected_toc[i] = corrected_item
                        break
        
        self.verified_toc = corrected_toc
        
        # Count how many items were successfully corrected
        still_incorrect = [item for item in corrected_toc if item.get("verification_status") == "incorrect"]
        
        self.processing_metadata["correction"] = {
            "items_to_correct": len(self.incorrect_items),
            "items_corrected": len(self.incorrect_items) - len(still_incorrect),
            "items_still_incorrect": len(still_incorrect)
        }
        
        self.logger.info({
            'event': 'toc_correction_complete',
            'items_to_correct': len(self.incorrect_items),
            'items_corrected': len(self.incorrect_items) - len(still_incorrect),
            'items_still_incorrect': len(still_incorrect)
        })
        
        return corrected_toc
    
    def _prepare_correction_data(self):
        """Prepare data for TOC correction"""
        correction_data = ""
        
        # For each incorrect item, get content from surrounding pages
        for item in self.incorrect_items:
            if item.get('physical_index') is not None:
                page_idx = item['physical_index']
                title = item['title']
                
                # Get content from the current page and adjacent pages
                for offset in range(-2, 3):  # -2, -1, 0, 1, 2
                    check_idx = page_idx + offset
                    if 0 <= check_idx < self.total_pages:
                        page_content = self.page_list[check_idx][0][:300]  # Get first 300 chars
                        correction_data += f"\n--- Page {check_idx} (Looking for: {title}) ---\n{page_content}\n"
        
        return correction_data
    
    # Just-in-time data preparation methods
    def get_sample_pages_for_analysis(self, num_pages: int = 5) -> List[str]:
        """Prepare sample pages for document analysis"""
        return [page[0][:1000] for page in self.page_list[:num_pages]]
    
    def get_pages_for_toc_detection(self, start_page: int = 0, max_pages: int = 10) -> List[str]:
        """Prepare pages for TOC detection"""
        end_page = min(start_page + max_pages, len(self.page_list))
        return [self.page_list[i][0] for i in range(start_page, end_page)]
    
    def get_grouped_content_for_toc_generation(self, max_tokens: int = 20000) -> List[str]:
        """Prepare grouped content with physical indices for TOC generation"""
        if self._grouped_content_cache is None:
            page_contents = []
            token_lengths = []
            
            for page_idx in range(1, len(self.page_list) + 1):
                page_text = f"<physical_index_{page_idx}>\n{self.page_list[page_idx-1][0]}\n<physical_index_{page_idx}>\n\n"
                page_contents.append(page_text)
                token_lengths.append(count_tokens(page_text, self.model))
            
            self._grouped_content_cache = page_list_to_group_text(page_contents, token_lengths, max_tokens)
        
        return self._grouped_content_cache
    
    def get_labeled_pages_for_mapping(self, start_page: int = 1, end_page: int = None) -> str:
        """Prepare labeled pages for page number mapping"""
        if end_page is None:
            end_page = len(self.page_list)
        
        cache_key = f"{start_page}_{end_page}"
        if not hasattr(self, '_labeled_cache'):
            self._labeled_cache = {}
        
        if cache_key not in self._labeled_cache:
            content_parts = []
            for page_idx in range(start_page, min(end_page + 1, len(self.page_list) + 1)):
                list_idx = page_idx - 1
                if 0 <= list_idx < len(self.page_list):
                    page_text = f"<physical_index_{page_idx}>\n{self.page_list[list_idx][0]}\n<physical_index_{page_idx}>\n\n"
                    content_parts.append(page_text)
            
            self._labeled_cache[cache_key] = ''.join(content_parts)
        
        return self._labeled_cache[cache_key]
    
    def get_verification_data_for_title(self, title: str, physical_index: int) -> Dict[str, Any]:
        """Prepare verification data for title checking"""
        if 1 <= physical_index <= len(self.page_list):
            page_content = self.page_list[physical_index - 1][0]
            return {
                "title": title,
                "physical_index": physical_index,
                "page_content": page_content
            }
        return {"title": title, "physical_index": physical_index, "page_content": ""}
    
    def get_content_for_toc_extraction(self, toc_pages: List[int]) -> str:
        """Prepare content from identified TOC pages"""
        content_parts = []
        for page_idx in toc_pages:
            if 0 <= page_idx < len(self.page_list):
                content_parts.append(self.page_list[page_idx][0])
        return "\n".join(content_parts)
    
    def get_section_content_for_summary(self, start_page: int, end_page: int) -> str:
        """Prepare section content for summary generation"""
        content_parts = []
        for page_idx in range(start_page, min(end_page + 1, len(self.page_list) + 1)):
            list_idx = page_idx - 1
            if 0 <= list_idx < len(self.page_list):
                content_parts.append(self.page_list[list_idx][0])
        return "\n".join(content_parts)
    
    # Agent result storage and retrieval
    def store_agent_result(self, agent_name: str, result: Any, metadata: Dict = None):
        """Store result from an agent for use by subsequent agents"""
        self.agent_results[agent_name] = {
            "result": result,
            "metadata": metadata or {},
            "timestamp": asyncio.get_event_loop().time()
        }
        
        self.processing_metadata["agents_used"].append(agent_name)
        self.logger.info(f"Stored result from {agent_name}")
    
    def get_agent_result(self, agent_name: str) -> Any:
        """Retrieve result from a previous agent"""
        return self.agent_results.get(agent_name, {}).get("result")
    
    def has_agent_result(self, agent_name: str) -> bool:
        """Check if agent has produced a result"""
        return agent_name in self.agent_results
    
    # Context state updates based on agent results
    def update_from_document_analysis(self, analysis_result: Dict):
        """Update context from document analysis"""
        if analysis_result:
            self.document_type = analysis_result.get("document_type", "unknown")
            self.complexity_level = analysis_result.get("complexity_level", "unknown")
            self.estimated_sections = analysis_result.get("estimated_sections", 0)
    
    def update_from_toc_detection(self, detection_result: Dict):
        """Update context from TOC detection"""
        if detection_result:
            self.has_toc = detection_result.get("has_toc", False)
            self.toc_pages = detection_result.get("toc_pages", [])
    
    def update_from_toc_extraction(self, extraction_result: str):
        """Update context from TOC extraction"""
        self.raw_toc_content = extraction_result or ""
    
    def update_from_toc_transformation(self, structured_toc: List[Dict]):
        """Update context from TOC transformation"""
        self.structured_toc = structured_toc or []
    
    def update_from_page_mapping(self, mapped_toc: List[Dict]):
        """Update context from page number mapping"""
        self.toc_with_physical_indices = mapped_toc or []
    
    def update_from_verification(self, verified_toc: List[Dict], incorrect_items: List[Dict]):
        """Update context from verification"""
        self.verified_toc = verified_toc or []
        self.incorrect_items = incorrect_items or []
    
    def get_current_state_summary(self) -> Dict[str, Any]:
        """Get summary of current processing state for orchestrator"""
        return {
            "document_characteristics": {
                "total_pages": self.total_pages,
                "document_type": self.document_type,
                "complexity_level": self.complexity_level,
                "estimated_sections": self.estimated_sections
            },
            "toc_status": {
                "has_toc": self.has_toc,
                "toc_pages_found": len(self.toc_pages),
                "toc_has_page_numbers": self.toc_has_page_numbers,
                "structured_toc_items": len(self.structured_toc),
                "verified_toc_items": len(self.verified_toc)
            },
            "processing_progress": {
                "agents_completed": len(self.agent_results),
                "agents_used": self.processing_metadata["agents_used"],
                "incorrect_items": len(self.incorrect_items)
            }
        }


class SpecializedAgent:
    """Wrapper for existing agents with context-aware data preparation"""
    
    def __init__(self, agent_name: str, agent_instance, responsibility: str):
        self.name = agent_name
        self.agent = agent_instance
        self.responsibility = responsibility
    
    async def execute(self, context: EnhancedPageIndexProcessingContext, 
                     task: AgentTask) -> AgentResult:
        """Execute agent with context-aware data preparation"""
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Prepare input based on agent type
            prompt = self._prepare_agent_prompt(context, task)
            
            # Execute agent
            raw_result = await run_specific_agent(self.agent, prompt)
            
            # Process result based on agent type
            processed_result = self._process_agent_result(raw_result, task.expected_output_type)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return AgentResult(
                agent_name=self.name,
                success=True,
                data=processed_result,
                confidence=0.8,  # Could be calculated based on result quality
                metadata={"execution_time": execution_time, "input_size": len(prompt)},
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            context.logger.error(f"Agent {self.name} failed: {str(e)}")
            
            return AgentResult(
                agent_name=self.name,
                success=False,
                data=None,
                confidence=0.0,
                metadata={"execution_time": execution_time},
                execution_time=execution_time,
                errors=[str(e)]
            )
    
    def _prepare_agent_prompt(self, context: EnhancedPageIndexProcessingContext, 
                            task: AgentTask) -> str:
        """Prepare agent-specific prompt with context data"""
        
        if self.name == "TOC_DETECTOR":
            pages = context.get_pages_for_toc_detection()
            return f"Analyze these pages for table of contents:\n\n" + "\n---\n".join(pages)
        
        elif self.name == "CREATE_TOC_FROM_CONTENT":
            grouped_content = context.get_grouped_content_for_toc_generation()
            chunk_idx = task.input_data.get("chunk_index", 0)
            total_chunks = len(grouped_content)
            
            if chunk_idx < len(grouped_content):
                return f"""Document Text Chunk {chunk_idx + 1}/{total_chunks}:
---
{grouped_content[chunk_idx]}
---
Please generate table of contents items from this text chunk according to your instructions.
If this is not the first chunk, assume it might be a continuation of a ToC from previous chunks."""
            
        elif self.name == "EXTRACT_TOC":
            toc_content = context.get_content_for_toc_extraction(context.toc_pages)
            return f"Extract the table of contents from:\n---\n{toc_content}\n---"
        
        elif self.name == "TOC_JSON_TRANSFORMER":
            raw_toc = task.input_data.get("raw_toc_content", context.raw_toc_content)
            return f"Transform this raw TOC to JSON:\n---\n{raw_toc}\n---"
        
        elif self.name == "ADD_PAGE_NUMBER":
            toc_structure = task.input_data.get("toc_structure", context.structured_toc)
            page_range = task.input_data.get("page_range", (1, context.total_pages))
            labeled_content = context.get_labeled_pages_for_mapping(page_range[0], page_range[1])
            
            return f"""Partial Document Text:
---
{labeled_content}
---
Given ToC Structure:
---
{json.dumps(toc_structure, indent=2)}
---
Map titles to physical page numbers."""
        
        elif self.name == "CHECK_TITLE_APPEARANCE":
            title = task.input_data.get("title")
            physical_index = task.input_data.get("physical_index")
            verification_data = context.get_verification_data_for_title(title, physical_index)
            
            return f"""Section Title: {title}
Page Text:
---
{verification_data['page_content']}
---
Check if the title appears in this page."""
        
        elif self.name == "SINGLE_TOC_ITEM_FIXER":
            title = task.input_data.get("title")
            search_range = task.input_data.get("search_range", (1, context.total_pages))
            search_content = context.get_labeled_pages_for_mapping(search_range[0], search_range[1])
            
            return f"""Section Title: {title}
Partial Document Text:
---
{search_content}
---
Find the correct physical page number for this title."""
        
        elif self.name == "NODE_SUMMARY":
            start_page = task.input_data.get("start_page")
            end_page = task.input_data.get("end_page")
            section_content = context.get_section_content_for_summary(start_page, end_page)
            
            return f"Generate summary for this section:\n---\n{section_content}\n---"
        
        elif self.name == "DOC_DESCRIPTION":
            return f"Generate description for document with structure:\n{json.dumps(context.final_structure, indent=2)}"
        
        # Add more agent-specific preparations as needed
        return task.input_data.get("prompt", "")
    
    def _process_agent_result(self, raw_result: str, expected_type: str) -> Any:
        """Process raw agent result based on expected type"""
        
        if expected_type == "json":
            return extract_json(raw_result)
        elif expected_type == "text":
            return raw_result.strip()
        elif expected_type == "boolean":
            # Extract yes/no from agent response
            result_json = extract_json(raw_result)
            if isinstance(result_json, dict):
                # Look for common boolean fields
                for field in ["has_toc", "complete", "answer", "verified"]:
                    if field in result_json:
                        return result_json[field] == "yes"
            return "yes" in raw_result.lower()
        else:
            return raw_result

# Multi-Agent Orchestrator
class MultiAgentOrchestrator:
    """Orchestrator that coordinates multiple specialized agents"""
    
    def __init__(self, context: EnhancedPageIndexProcessingContext):
        self.context = context
        self.logger = context.logger
        
        # Initialize specialized agents
        self.agents = {
            "document_analyzer": SpecializedAgent(DOCUMENT_ANALYZER_AGENT, "document_analyzer"),
            "toc_detector": SpecializedAgent(TOC_DETECTOR_AGENT, "toc_detector"),
            "toc_extractor": SpecializedAgent(EXTRACT_TOC_AGENT, "toc_extractor"),
            "page_index_detector": SpecializedAgent(PAGE_INDEX_DETECTOR_AGENT, "page_index_detector"),
            "toc_transformer": SpecializedAgent(TOC_JSON_TRANSFORMER_AGENT, "toc_transformer"),
            "page_number_mapper": SpecializedAgent(ADD_PAGE_NUMBER_AGENT, "page_number_mapper"),
            "title_verifier": SpecializedAgent(VERIFY_TOC_AGENT, "title_verifier"),
            "toc_corrector": SpecializedAgent(FIX_TOC_AGENT, "toc_corrector"),
            "doc_describer": SpecializedAgent(DOC_DESCRIPTION_AGENT, "doc_describer"),
            "toc_extraction_complete_checker": SpecializedAgent(TOC_EXTRACTION_COMPLETE_AGENT, "toc_extraction_complete_checker"),
            "toc_transformation_complete_checker": SpecializedAgent(TOC_TRANSFORMATION_COMPLETE_AGENT, "toc_transformation_complete_checker"),
        }
        
        # Initialize orchestrator agent
        self.orchestrator_agent = Agent(
            name="MultiAgentOrchestrator",
            instructions="""You are a multi-agent orchestrator for document page indexing.

Your role is to:
1. Analyze current context and processing state
2. Determine the next best agent to execute
3. Decide on strategy adaptations based on intermediate results
4. Coordinate agent execution for optimal results

Available agents and their capabilities:
- TOC_DETECTOR: Detect presence of table of contents
- EXTRACT_TOC: Extract raw TOC content from pages  
- TOC_JSON_TRANSFORMER: Convert raw TOC to structured JSON
- CREATE_TOC_FROM_CONTENT: Generate TOC from document content analysis
- ADD_PAGE_NUMBER: Map TOC titles to physical page numbers
- CHECK_TITLE_APPEARANCE: Verify title locations on pages
- SINGLE_TOC_ITEM_FIXER: Fix incorrect page mappings
- NODE_SUMMARY: Generate section summaries
- DOC_DESCRIPTION: Generate document description

Respond with JSON containing your decision:
{
    "next_agent": "agent_name",
    "task_parameters": {"key": "value"},
    "reasoning": "explanation of choice",
    "alternative_agents": ["backup options"],
    "completion_status": "continue|complete|adapt_strategy",
    "quality_assessment": 0.0-1.0
}
""",
            model="gpt-4.1"
        )
    
    def _initialize_agents(self) -> Dict[str, SpecializedAgent]:
        """Initialize all specialized agents"""
        return {
            "TOC_DETECTOR": SpecializedAgent("TOC_DETECTOR", TOC_DETECTOR_AGENT, 
                                           "Detect table of contents presence"),
            "EXTRACT_TOC": SpecializedAgent("EXTRACT_TOC", EXTRACT_TOC_AGENT,
                                          "Extract raw TOC content"),
            "TOC_JSON_TRANSFORMER": SpecializedAgent("TOC_JSON_TRANSFORMER", TOC_JSON_TRANSFORMER_AGENT,
                                                   "Transform TOC to JSON structure"),
            "CREATE_TOC_FROM_CONTENT": SpecializedAgent("CREATE_TOC_FROM_CONTENT", CREATE_TOC_FROM_CONTENT_AGENT,
                                                      "Generate TOC from content analysis"),
            "ADD_PAGE_NUMBER": SpecializedAgent("ADD_PAGE_NUMBER", ADD_PAGE_NUMBER_AGENT,
                                              "Map titles to physical pages"),
            "CHECK_TITLE_APPEARANCE": SpecializedAgent("CHECK_TITLE_APPEARANCE", CHECK_TITLE_APPEARANCE_AGENT,
                                                     "Verify title locations"),
            "SINGLE_TOC_ITEM_FIXER": SpecializedAgent("SINGLE_TOC_ITEM_FIXER", SINGLE_TOC_ITEM_FIXER_AGENT,
                                                    "Fix incorrect page mappings"),
            "NODE_SUMMARY": SpecializedAgent("NODE_SUMMARY", NODE_SUMMARY_AGENT,
                                           "Generate section summaries"),
            "DOC_DESCRIPTION": SpecializedAgent("DOC_DESCRIPTION", DOC_DESCRIPTION_AGENT,
                                              "Generate document description"),
            "PAGE_INDEX_DETECTOR": SpecializedAgent("PAGE_INDEX_DETECTOR", PAGE_INDEX_DETECTOR_AGENT,
                                                  "Detect page numbers in TOC")
        }
    
    async def orchestrate_processing(self) -> Dict[str, Any]:
        """Main orchestration loop with enhanced TOC detection and processing"""
        
        start_time = asyncio.get_event_loop().time()
        self.context.logger.info({
            'event': 'orchestration_started',
            'timestamp': start_time
        })
        
        # Phase 1: Document Analysis
        await self._perform_document_analysis()
        
        # Phase 2: TOC Detection and Processing
        await self._process_toc()
        
        # Phase 3: Verification and Correction
        if self.context.toc_with_physical_indices:
            await self._verify_and_correct_toc()
        
        # Phase 4: Finalization
        await self._finalize_processing()
        
        end_time = asyncio.get_event_loop().time()
        self.context.logger.info({
            'event': 'orchestration_completed',
            'total_execution_time': end_time - start_time,
            'agents_used': len(self.context.processing_metadata["agents_used"]),
            'strategy_adaptations': len(self.context.processing_metadata["strategy_adaptations"])
        })
        
        return self.context.get_current_state_summary()
    
    async def _perform_document_analysis(self):
        """Analyze document characteristics"""
        self.context.logger.info({'event': 'document_analysis_started'})
        
        # Get sample pages for analysis
        sample_pages = self.context.get_sample_pages_for_analysis(5)
        
        # Create task for document analyzer
        task = AgentTask(
            agent_name="document_analyzer",
            input_data={"sample_pages": sample_pages},
            expected_output_type="json"
        )
        
        # Execute document analyzer
        result = await self.agents["document_analyzer"].execute(self.context, task)
        
        # Update context with analysis results
        if result.success and isinstance(result.data, dict):
            self.context.update_from_document_analysis(result.data)
            
            # Determine processing strategy based on document type
            if result.data.get("document_type") in ["academic", "technical", "report"]:
                self.context.processing_strategy = "toc_with_page_numbers"
            elif result.data.get("document_type") in ["presentation", "slides"]:
                self.context.processing_strategy = "no_toc"
            
            self.context.logger.info({
                'event': 'document_analysis_complete',
                'document_type': self.context.document_type,
                'complexity_level': self.context.complexity_level,
                'processing_strategy': self.context.processing_strategy
            })
        else:
            self.context.logger.warning({
                'event': 'document_analysis_failed',
                'error': str(result.errors) if result.errors else "Unknown error"
            })
    
    async def _process_toc(self):
        """Process table of contents using enhanced methods"""
        self.context.logger.info({'event': 'toc_processing_started'})
        
        # Step 1: Find TOC pages
        toc_pages = await self.context.find_toc_pages(0, 10)
        
        if not self.context.has_toc:
            self.context.logger.info({'event': 'no_toc_found', 'strategy': 'generate_from_content'})
            await self._generate_toc_from_content()
            return
        
        # Step 2: Extract TOC content from identified pages
        toc_content = self.context.get_content_for_toc_extraction(toc_pages)
        
        task = AgentTask(
            agent_name="toc_extractor",
            input_data={"toc_content": toc_content},
            expected_output_type="text"
        )
        
        result = await self.agents["toc_extractor"].execute(self.context, task)
        
        if result.success and result.data:
            self.context.raw_toc_content = result.data
            self.context.logger.info({
                'event': 'toc_extraction_complete',
                'content_length': len(result.data)
            })
        else:
            self.context.logger.warning({'event': 'toc_extraction_failed'})
            await self._generate_toc_from_content()
            return
        
        # Step 3: Detect page numbers in TOC
        has_page_numbers = await self.context.detect_page_numbers_in_toc()
        
        # Step 4: Transform TOC to structured format
        structured_toc = await self.context.transform_toc_to_json()
        
        if not structured_toc:
            self.context.logger.warning({'event': 'toc_transformation_failed'})
            await self._generate_toc_from_content()
            return
        
        # Step 5: Add page numbers to TOC items
        mapped_toc = await self.context.add_page_numbers_to_toc()
        
        # Step 6: Validate and truncate physical indices
        await self.context.validate_and_truncate_physical_indices()
        
        self.context.logger.info({
            'event': 'toc_processing_complete',
            'items_with_physical_index': sum(1 for item in self.context.toc_with_physical_indices 
                                         if item.get('physical_index') is not None)
        })
    
    async def _generate_toc_from_content(self):
        """Generate TOC from document content when no TOC is found"""
        self.context.logger.info({'event': 'generating_toc_from_content'})
        
        # Get grouped content for TOC generation
        grouped_content = self.context.get_grouped_content_for_toc_generation()
        
        generated_toc = []
        
        # Process each content chunk
        for i, content_chunk in enumerate(grouped_content):
            task = AgentTask(
                agent_name="toc_transformer",
                input_data={
                    "content": content_chunk,
                    "chunk_index": i,
                    "total_chunks": len(grouped_content)
                },
                expected_output_type="json"
            )
            
            result = await self.agents["toc_transformer"].execute(self.context, task)
            
            if result.success and isinstance(result.data, dict) and "table_of_contents" in result.data:
                generated_toc.extend(result.data["table_of_contents"])
        
        # Update context with generated TOC
        if generated_toc:
            self.context.structured_toc = generated_toc
            self.context.toc_transformation_complete = True
            
            # Add page numbers to generated TOC
            await self.context.add_page_numbers_to_toc()
            await self.context.validate_and_truncate_physical_indices()
            
            self.context.logger.info({
                'event': 'toc_generation_complete',
                'generated_items': len(generated_toc)
            })
        else:
            self.context.logger.warning({'event': 'toc_generation_failed'})
    
    async def _verify_and_correct_toc(self):
        """Verify and correct TOC entries"""
        self.context.logger.info({'event': 'toc_verification_started'})
        
        # Step 1: Verify TOC entries
        verified_toc = await self.context.verify_toc()
        
        # Step 2: Fix incorrect TOC entries if any
        if self.context.incorrect_items:
            self.context.logger.info({
                'event': 'fixing_incorrect_toc_items',
                'incorrect_items': len(self.context.incorrect_items)
            })
            
            corrected_toc = await self.context.fix_incorrect_toc()
            
            # If there are still incorrect items, try one more time
            still_incorrect = [item for item in corrected_toc if item.get("verification_status") == "incorrect"]
            if still_incorrect:
                self.context.logger.info({'event': 'second_correction_attempt'})
                await self.context.fix_incorrect_toc()
        
        self.context.logger.info({'event': 'toc_verification_complete'})

    
    async def _get_orchestrator_decision(self) -> Dict[str, Any]:
        """Get decision from orchestrator agent"""
        
        context_summary = self.context.get_current_state_summary()
        
        decision_prompt = f"""
        Current processing state:
        {json.dumps(context_summary, indent=2)}
        
        Agent execution history:
        {self.context.processing_metadata["agents_used"]}
        
        Available agent results:
        {list(self.context.agent_results.keys())}
        
        Determine the next action based on current state and progress.
        """
        
        response = await run_specific_agent(self.orchestrator_agent, decision_prompt)
        decision = extract_json(response)
        
        # Fallback decision if orchestrator fails
        if not decision:
            return self._get_fallback_decision()
        
        return decision
    
    def _get_fallback_decision(self) -> Dict[str, Any]:
        """Provide fallback decision when orchestrator fails"""
        
        # Simple rule-based fallback logic
        if not self.context.has_agent_result("TOC_DETECTOR"):
            return {
                "next_agent": "TOC_DETECTOR",
                "task_parameters": {"expected_output": "boolean"},
                "reasoning": "Start with TOC detection",
                "completion_status": "continue"
            }
        elif self.context.has_toc and not self.context.has_agent_result("EXTRACT_TOC"):
            return {
                "next_agent": "EXTRACT_TOC",
                "task_parameters": {"expected_output": "text"},
                "reasoning": "Extract TOC content",
                "completion_status": "continue"
            }
        elif not self.context.has_toc and not self.context.has_agent_result("CREATE_TOC_FROM_CONTENT"):
            return {
                "next_agent": "CREATE_TOC_FROM_CONTENT", 
                "task_parameters": {"chunk_index": 0, "expected_output": "json"},
                "reasoning": "Generate TOC from content",
                "completion_status": "continue"
            }
        else:
            return {"completion_status": "complete"}
    
    def _update_context_from_agent_result(self, agent_name: str, result: AgentResult):
        """Update context based on agent results"""
        
        if not result.success:
            return
        
        if agent_name == "TOC_DETECTOR":
            self.context.update_from_toc_detection({
                "has_toc": result.data,
                "toc_pages": []  # Would be populated by actual detection logic
            })
        
        elif agent_name == "EXTRACT_TOC":
            self.context.update_from_toc_extraction(result.data)
        
        elif agent_name == "TOC_JSON_TRANSFORMER":
            if isinstance(result.data, dict) and "table_of_contents" in result.data:
                self.context.update_from_toc_transformation(result.data["table_of_contents"])
        
        elif agent_name == "CREATE_TOC_FROM_CONTENT":
            if isinstance(result.data, list):
                # Accumulate results from multiple content chunks
                existing_toc = self.context.structured_toc or []
                existing_toc.extend(result.data)
                self.context.update_from_toc_transformation(existing_toc)
        
        elif agent_name == "ADD_PAGE_NUMBER":
            if isinstance(result.data, list):
                self.context.update_from_page_mapping(result.data)
    
    async def _adapt_strategy(self, decision: Dict[str, Any]):
        """Adapt processing strategy based on orchestrator decision"""
        
        adaptation_reason = decision.get("reasoning", "Unknown reason")
        self.context.processing_metadata["strategy_adaptations"].append({
            "reason": adaptation_reason,
            "timestamp": asyncio.get_event_loop().time(),
            "context_state": self.context.get_current_state_summary()
        })
        
        self.context.logger.info(f"Strategy adaptation: {adaptation_reason}")
    
    async def _finalize_processing(self):
        """Finalize processing and build final structure with enhanced post-processing"""
        self.context.logger.info({'event': 'finalization_started'})
        
        # Step 1: Select the best TOC representation available
        if self.context.verified_toc:
            toc_items = self.context.verified_toc
            source = "verified_toc"
        elif self.context.toc_with_physical_indices:
            toc_items = self.context.toc_with_physical_indices
            source = "toc_with_physical_indices"
        elif self.context.structured_toc:
            # Convert physical indices if needed
            toc_items = self.context.structured_toc
            source = "structured_toc"
        else:
            toc_items = []
            source = "empty"
        
        self.context.logger.info({
            'event': 'toc_source_selected',
            'source': source,
            'items_count': len(toc_items)
        })
        
        # Step 2: Filter valid items with physical indices
        valid_items = [item for item in toc_items if item.get('physical_index') is not None]
        
        # Step 3: Process large nodes recursively if needed
        processed_items = await self._process_large_nodes_recursively(valid_items)
        
        # Step 4: Apply post-processing to build final structure
        if processed_items:
            # Apply post-processing to build hierarchical structure
            final_structure = self._apply_post_processing(processed_items)
            self.context.final_structure = final_structure
            
            # Apply meta-processing for additional attributes
            await self._apply_meta_processing()
            
            self.context.logger.info({
                'event': 'finalization_complete',
                'final_structure_nodes': len(final_structure),
                'max_depth': self._calculate_max_depth(final_structure)
            })
        else:
            self.context.logger.warning({'event': 'finalization_failed', 'reason': 'no_valid_items'})
    
    async def _process_large_nodes_recursively(self, toc_items):
        """Process large nodes recursively to improve structure"""
        if not toc_items:
            return []
            
        # Identify large nodes (sections with many pages)
        large_nodes = []
        for i, item in enumerate(toc_items):
            if i < len(toc_items) - 1:
                current_index = item.get('physical_index')
                next_index = toc_items[i+1].get('physical_index')
                
                if current_index and next_index and (next_index - current_index) > 20:
                    large_nodes.append(item)
        
        # Process large nodes if found
        if large_nodes:
            self.context.logger.info({
                'event': 'large_nodes_detected',
                'count': len(large_nodes)
            })
            
            for node in large_nodes:
                await self._process_single_large_node(node, toc_items)
        
        return toc_items
    
    async def _process_single_large_node(self, node, all_items):
        """Process a single large node to find subsections"""
        # Find node index in items list
        node_idx = next((i for i, item in enumerate(all_items) if item == node), -1)
        if node_idx == -1 or node_idx >= len(all_items) - 1:
            return
            
        # Get start and end page for this section
        start_page = node.get('physical_index')
        end_page = all_items[node_idx + 1].get('physical_index')
        
        if not start_page or not end_page or end_page - start_page <= 20:
            return
            
        # Get content for this section
        section_content = self.context.get_section_content_for_summary(start_page, end_page)
        
        # Try to identify subsections
        task = AgentTask(
            agent_name="toc_transformer",
            input_data={
                "content": section_content,
                "parent_section": node['title'],
                "start_page": start_page,
                "end_page": end_page
            },
            expected_output_type="json"
        )
        
        result = await self.agents["toc_transformer"].execute(self.context, task)
        
        if result.success and isinstance(result.data, dict) and "table_of_contents" in result.data:
            subsections = result.data["table_of_contents"]
            
            # Add subsections to the main TOC items
            if subsections:
                # Set parent reference
                for subsection in subsections:
                    subsection['parent'] = node['title']
                    subsection['level'] = node.get('level', 0) + 1
                
                # Insert subsections after the parent node
                all_items[node_idx+1:node_idx+1] = subsections
                
                self.context.logger.info({
                    'event': 'subsections_added',
                    'parent': node['title'],
                    'subsections_count': len(subsections)
                })
    
    def _apply_post_processing(self, toc_items):
        """Apply post-processing to build hierarchical structure"""
        # Sort by physical index
        sorted_items = sorted(toc_items, key=lambda x: x.get('physical_index', 0))
        
        # Build hierarchical structure
        hierarchical_structure = []
        current_path = []
        
        for item in sorted_items:
            level = item.get('level', 0)
            
            # Adjust current path based on level
            while len(current_path) > level:
                current_path.pop()
            
            # Create node
            node = {
                'title': item.get('title', ''),
                'physical_index': item.get('physical_index'),
                'children': []
            }
            
            # Add to parent or root
            if current_path:
                current_path[-1]['children'].append(node)
            else:
                hierarchical_structure.append(node)
            
            # Update current path
            if level >= len(current_path):
                current_path.append(node)
        
        return hierarchical_structure
    
    async def _apply_meta_processing(self):
        """Apply meta-processing to add additional attributes"""
        # Add node IDs
        self._add_node_ids(self.context.final_structure)
        
        # Add page ranges
        self._add_page_ranges(self.context.final_structure)
        
        # Add document metadata
        self.context.final_structure = {
            'title': self.context.doc_name,
            'document_type': self.context.document_type,
            'total_pages': self.context.total_pages,
            'has_toc': self.context.has_toc,
            'sections': self.context.final_structure
        }
    
    def _add_node_ids(self, nodes, prefix=''):
        """Add unique IDs to nodes"""
        for i, node in enumerate(nodes):
            node_id = f"{prefix}{i+1}"
            node['id'] = node_id
            
            if 'children' in node and node['children']:
                self._add_node_ids(node['children'], f"{node_id}.") 
    
    def _add_page_ranges(self, nodes):
        """Add page ranges to nodes"""
        for i, node in enumerate(nodes):
            start_page = node.get('physical_index')
            
            # Determine end page
            if i < len(nodes) - 1:
                end_page = nodes[i+1].get('physical_index', self.context.total_pages) - 1
            else:
                end_page = self.context.total_pages
            
            node['page_range'] = {'start': start_page, 'end': end_page}
            
            # Process children recursively
            if 'children' in node and node['children']:
                self._add_page_ranges(node['children'])
    
    def _calculate_max_depth(self, nodes, current_depth=1):
        """Calculate maximum depth of the structure"""
        if not nodes:
            return current_depth - 1
            
        max_child_depth = current_depth
        for node in nodes:
            if 'children' in node and node['children']:
                child_depth = self._calculate_max_depth(node['children'], current_depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
                
        return max_child_depth

# Main function using multi-agent orchestrator
async def multi_agent_page_index(doc, opt=None, logger=None, model=None):
    """Main function using multi-agent orchestrator with context"""
    
    try:
        # Initialize enhanced context
        context = EnhancedPageIndexProcessingContext(doc, opt, logger, model)
        await context.initialize()
        
        # Create and run orchestrator
        orchestrator = MultiAgentOrchestrator(context)
        processing_summary = await orchestrator.orchestrate_processing()
        
        # Apply enhancements if requested
        if context.opt.include_node_id == 'yes':
            from pageindex.utils import write_node_id
            write_node_id(context.final_structure)
        
        if context.opt.include_node_text == 'yes':
            from pageindex.utils import add_node_text
            add_node_text(context.final_structure, context.page_list)
        
        # Generate summaries and description if requested
        if context.opt.include_node_summary == 'yes':
            summary_agent = orchestrator.agents["NODE_SUMMARY"]
            # Implementation for generating summaries for each node
        
        if context.opt.include_doc_description == 'yes':
            description_agent = orchestrator.agents["DOC_DESCRIPTION"]
            # Implementation for generating document description
        
        # Return final result
        return {
            'doc_name': context.doc_name,
            'structure': context.final_structure,
            'processing_metadata': {
                'agents_used': context.processing_metadata["agents_used"],
                'strategy_adaptations': context.processing_metadata["strategy_adaptations"],
                'total_execution_time': sum(r["metadata"].get("execution_time", 0) 
                                          for r in context.agent_results.values()),
                'processing_summary': processing_summary
            }
        }
    except Exception as e:
        if logger:
            logger.error(f"Error in multi_agent_page_index: {str(e)}")
        else:
            print(f"Error in multi_agent_page_index: {str(e)}")
        raise