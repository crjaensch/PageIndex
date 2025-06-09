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
        self.total_pages = 0
        self.total_tokens = 0
        self.document_type = "unknown"
        self.complexity_level = "unknown"
        self.estimated_sections = 0
        
        # TOC-related data (populated by detection agents)
        self.has_toc = False
        self.toc_pages: List[int] = []
        self.raw_toc_content = ""
        self.toc_has_page_numbers = False
        self.structured_toc: List[Dict] = []
        
        # Processing results (populated by processing agents)
        self.toc_with_physical_indices: List[Dict] = []
        self.verified_toc: List[Dict] = []
        self.final_structure: List[Dict] = []
        self.incorrect_items: List[Dict] = []
        
        # Enhancement data (populated by enhancement agents)
        self.node_summaries: Dict[str, str] = {}
        self.document_description = ""
        
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
            "quality_scores": {},
            "execution_timeline": []
        }
    
    async def initialize(self):
        """Initialize basic document data"""
        self.page_list = get_page_tokens(self.doc)
        self.total_pages = len(self.page_list)
        self.total_tokens = sum(page[1] for page in self.page_list)
        
        self.logger.info({
            'total_pages': self.total_pages,
            'total_tokens': self.total_tokens
        })
        return self
    
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
    """Orchestrator that coordinates multiple agents using context"""
    
    def __init__(self, context: EnhancedPageIndexProcessingContext):
        self.context = context
        self.agents = self._initialize_agents()
        
        # Main orchestrator agent for strategy decisions
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
        """Main orchestration loop"""
        
        self.context.logger.info("Starting multi-agent orchestration")
        
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Get orchestrator decision
            decision = await self._get_orchestrator_decision()
            
            if decision.get("completion_status") == "complete":
                self.context.logger.info("Orchestrator determined processing is complete")
                break
            elif decision.get("completion_status") == "adapt_strategy":
                self.context.logger.info("Orchestrator requested strategy adaptation")
                await self._adapt_strategy(decision)
                continue
            
            # Execute next agent
            agent_name = decision.get("next_agent")
            if agent_name and agent_name in self.agents:
                task_params = decision.get("task_parameters", {})
                
                task = AgentTask(
                    agent_name=agent_name,
                    input_data=task_params,
                    expected_output_type=task_params.get("expected_output", "json")
                )
                
                result = await self.agents[agent_name].execute(self.context, task)
                
                # Update context with agent result
                self._update_context_from_agent_result(agent_name, result)
                
                # Store result for potential use by other agents
                self.context.store_agent_result(agent_name, result.data, result.metadata)
                
                self.context.logger.info(f"Completed agent {agent_name} with success: {result.success}")
            
            else:
                self.context.logger.warning(f"Unknown agent requested: {agent_name}")
                break
        
        # Finalize processing
        await self._finalize_processing()
        
        return self.context.get_current_state_summary()
    
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
        """Finalize processing and build final structure"""
        
        # Build final structure from verified TOC
        if self.context.verified_toc:
            toc_items = self.context.verified_toc
        elif self.context.toc_with_physical_indices:
            toc_items = self.context.toc_with_physical_indices
        elif self.context.structured_toc:
            # Convert physical indices if needed
            toc_items = convert_physical_index_to_int(self.context.structured_toc)
        else:
            toc_items = []
        
        # Filter valid items and build structure
        valid_items = [item for item in toc_items if item.get('physical_index') is not None]
        
        if valid_items:
            self.context.final_structure = post_processing(valid_items, self.context.total_pages)
        
        self.context.logger.info("Processing finalized")

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