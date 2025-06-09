import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from agents import Agent
from pageindex.utils import run_specific_agent, extract_json, count_tokens, JsonLogger

class TaskType(Enum):
    DOCUMENT_ANALYSIS = "document_analysis"
    TOC_DETECTION = "toc_detection"
    TOC_EXTRACTION = "toc_extraction"
    CONTENT_STRUCTURING = "content_structuring"
    PAGE_MAPPING = "page_mapping"
    VERIFICATION = "verification"
    CORRECTION = "correction"
    ENHANCEMENT = "enhancement"

class AgentCapability(Enum):
    TOC_DETECTION = "toc_detection"
    TOC_EXTRACTION = "toc_extraction"
    PAGE_INDEX_DETECTION = "page_index_detection"
    JSON_TRANSFORMATION = "json_transformation"
    CONTENT_TOC_GENERATION = "content_toc_generation"
    PAGE_NUMBER_MAPPING = "page_number_mapping"
    TITLE_VERIFICATION = "title_verification"
    ITEM_FIXING = "item_fixing"
    TITLE_START_CHECKING = "title_start_checking"
    SUMMARY_GENERATION = "summary_generation"
    DESCRIPTION_GENERATION = "description_generation"

@dataclass
class AgentInfo:
    name: str
    agent: Agent
    capabilities: List[AgentCapability]
    complexity_level: int  # 1-5, where 5 is most complex
    reliability_score: float  # 0-1, based on historical performance
    token_efficiency: float  # tokens per operation
    description: str

@dataclass
class DocumentContext:
    total_pages: int
    total_tokens: int
    avg_tokens_per_page: float
    has_complex_structure: bool
    estimated_sections: int
    language: str = "unknown"
    document_type: str = "unknown"  # academic, technical, legal, etc.

@dataclass
class TaskResult:
    success: bool
    data: Any
    confidence: float
    tokens_used: int
    execution_time: float
    errors: List[str]
    agent_used: str

@dataclass
class StrategyPlan:
    strategy_name: str
    agent_sequence: List[str]
    fallback_strategies: List[str]
    estimated_confidence: float
    estimated_tokens: int
    reasoning: str

class AgentRegistry:
    """Registry of all available agents with their capabilities"""
    
    def __init__(self):
        self.agents: Dict[str, AgentInfo] = {
            "toc_detector": AgentInfo(
                name="TOC_DETECTOR_AGENT",
                agent=Agent(
                    name="TocDetectorAgent",
                    instructions="""Your job is to detect if there is a table of content provided in the given text. 
Note that abstract, summary, notation list, figure list, table list, etc., are not table of contents. 
You must reply in the specified JSON format: {\\\"thinking\\\": \\\"<why you think there is a ToC>\\\", 
\\\"toc_detected\\\": \\\"yes or no\\\"}. Directly return the final JSON structure. Do not output anything else.
""",
                    model="gpt-4.1-mini"
                ),
                capabilities=[AgentCapability.TOC_DETECTION],
                complexity_level=2,
                reliability_score=0.92,
                token_efficiency=0.8,
                description="Detects presence and type of table of contents in documents"
            ),
            
            "toc_extractor": AgentInfo(
                name="TOC_EXTRACTOR_AGENT", 
                agent=Agent(
                    name="TocExtractorAgent",
                    instructions="""Your job is to extract the full table of contents (ToC) from the given text.
Replace '...' with ':' if appropriate. Ensure all relevant sections are included. Directly return the full 
table of contents as a single block of text. Do not output anything else.
""",
                    model="gpt-4.1-mini"
                ),
                capabilities=[AgentCapability.TOC_EXTRACTION],
                complexity_level=3,
                reliability_score=0.88,
                token_efficiency=0.7,
                description="Extracts and cleans table of contents text"
            ),
            
            "page_index_detector": AgentInfo(
                name="PAGE_INDEX_DETECTOR_AGENT",
                agent=Agent(
                    name="PageIndexDetectorAgent", 
                    instructions="""Analyze TOC text to determine if page numbers are present.
                    Return JSON: {{"has_page_numbers": boolean, "confidence": float, "number_format": string}}""",
                    model="gpt-4.1-mini"
                ),
                capabilities=[AgentCapability.PAGE_INDEX_DETECTION],
                complexity_level=2,
                reliability_score=0.94,
                token_efficiency=0.9,
                description="Detects presence of page numbers in table of contents"
            ),
            
            "json_transformer": AgentInfo(
                name="JSON_TRANSFORMER_AGENT",
                agent=Agent(
                    name="JsonTransformerAgent",
                    instructions="""You are an expert in parsing and structuring table of contents (ToC) data.
You will be given raw text representing a table of contents.
Your task is to transform this entire raw ToC text into a specific JSON format.
The required JSON output structure is:
{
  "table_of_contents": [
    {
      "structure": "<structure index string, e.g., '1.1.2', or null if not applicable/present>",
      "title": "<title of the section as a string>",
      "page": "<page number as an integer, or null if not applicable/present>"
    }
    // ... additional items for each entry in the ToC
  ]
}
- The 'structure' field is a string representing the hierarchical index (e.g., "1", "1.1", "A.2.c"). If no structure/numbering is present for an item, use null.
- The 'title' field is the verbatim title of the section.
- The 'page' field is the page number associated with the title. If no page number is present for an item, use null.
Ensure you process the *entire* provided table of contents text and transform it completely in one go.
Directly return *only* the final JSON object (starting with '{' and ending with '}'). Do not add any explanatory text before or after the JSON.
""",
                    model="gpt-4.1-mini"
                ),
                capabilities=[AgentCapability.JSON_TRANSFORMATION],
                complexity_level=4,
                reliability_score=0.85,
                token_efficiency=0.6,
                description="Converts unstructured TOC to structured JSON"
            ),
            
            "content_toc_generator": AgentInfo(
                name="CONTENT_TOC_GENERATOR_AGENT",
                agent=Agent(
                    name="ContentTocGeneratorAgent",
                    instructions="""You are an expert in analyzing document text and generating a hierarchical table of contents (ToC) from it.
Given a chunk of document text, which may include <physical_index_X> tags indicating page numbers:
1. Identify section titles within the text.
2. Determine their hierarchical structure (e.g., 1, 1.1, 1.2, 2).
3. Extract their corresponding physical page numbers (from the <physical_index_X> tags nearest to the start of each title).
4. Return a JSON list of ToC items. Each item in the list must be an object with the following keys:
    - "structure": (string) The hierarchical structure index (e.g., "1", "1.1").
    - "title": (string) The extracted section title.
    - "physical_index": (string) The <physical_index_X> tag (e.g., "<physical_index_123>"). If no specific index tag is clearly associated with a title in the provided text, this can be null, but strive to find it.

Important:
- Focus only on the provided text chunk.
- Ensure the output is a valid JSON list of objects.
- If the text chunk appears to be a continuation of a ToC from a previous chunk, try to make your structural numbering logical in that context if possible, but primarily focus on the content of the current chunk.
- Directly return *only* the final JSON list. Do not add any explanatory text before or after the JSON.
""",
                    model="gpt-4.1-mini"
                ),
                capabilities=[AgentCapability.CONTENT_TOC_GENERATION],
                complexity_level=5,
                reliability_score=0.78,
                token_efficiency=0.4,
                description="Generates TOC structure from document content analysis"
            ),
            
            "page_mapper": AgentInfo(
                name="PAGE_MAPPER_AGENT",
                agent=Agent(
                    name="PageMapperAgent",
                    instructions="""Map section titles to their physical page locations.
                    Use document text with page markers to find accurate page numbers.
                    Return updated JSON with physical_index fields populated.""",
                    model="gpt-4.1-mini"
                ),
                capabilities=[AgentCapability.PAGE_NUMBER_MAPPING],
                complexity_level=4,
                reliability_score=0.82,
                token_efficiency=0.5,
                description="Maps section titles to physical page numbers"
            ),
            
            "title_verifier": AgentInfo(
                name="TITLE_VERIFIER_AGENT",
                agent=Agent(
                    name="TitleVerifierAgent",
                    instructions="""Your job is to check if a given section title appears or starts 
in the given page_text. Use fuzzy matching and ignore space inconsistencies. You must reply in the 
specified JSON format: {{\"thinking\": \"<your reasoning>\", \"answer\": \"yes or no\"}}. 
Directly return the final JSON structure. Do not output anything else.
""",
                    model="gpt-4.1-mini"
                ),
                capabilities=[AgentCapability.TITLE_VERIFICATION],
                complexity_level=3,
                reliability_score=0.91,
                token_efficiency=0.7,
                description="Verifies title locations on specified pages"
            ),
            
            "item_fixer": AgentInfo(
                name="ITEM_FIXER_AGENT",
                agent=Agent(
                    name="ItemFixerAgent",
                    instructions="""You are an expert in finding the physical page number for a single 
table of contents (ToC) item title within a given text snippet. Given a "Section Title" and "Partial Document Text" 
(which includes <physical_index_X> tags):
1. Locate the first occurrence of the "Section Title" in the "Partial Document Text".
2. Identify the <physical_index_X> tag that corresponds to the page where this title starts.
3. Return a JSON object containing only the "physical_index" as a string in the format "<physical_index_X>".

Example Input:
Section Title: "Chapter 1: Introduction"
Partial Document Text:
---
<physical_index_5>
Some text...
Chapter 1: Introduction
More text...
<physical_index_5>
<physical_index_6>
...
<physical_index_6>
---

Example Output:
{
    "physical_index": "<physical_index_5>"
}

If the title is not found, or if no physical_index tag is associated, return:
{
    "physical_index": null
}
Directly return *only* the final JSON object. Do not add any explanatory text.
""",
                    model="gpt-4.1-mini"
                ),
                capabilities=[AgentCapability.ITEM_FIXING],
                complexity_level=4,
                reliability_score=0.79,
                token_efficiency=0.6,
                description="Corrects individual TOC item page mappings"
            ),
            
            "summary_generator": AgentInfo(
                name="SUMMARY_GENERATOR_AGENT",
                agent=Agent(
                    name="SummaryGeneratorAgent",
                    instructions="""Generate concise summaries for document sections.
                    Focus on main points and key information in each section.""",
                    model="gpt-4.1-mini"
                ),
                capabilities=[AgentCapability.SUMMARY_GENERATION],
                complexity_level=3,
                reliability_score=0.87,
                token_efficiency=0.5,
                description="Generates section summaries"
            ),
            
            "doc_descriptor": AgentInfo(
                name="DOC_DESCRIPTOR_AGENT",
                agent=Agent(
                    name="DocDescriptorAgent",
                    instructions="""Generate comprehensive document description.
                    Analyze structure and content to create distinguishing description.""",
                    model="gpt-4.1-mini"
                ),
                capabilities=[AgentCapability.DESCRIPTION_GENERATION],
                complexity_level=3,
                reliability_score=0.89,
                token_efficiency=0.6,
                description="Generates document-level descriptions"
            )
        }
    
    def get_agents_by_capability(self, capability: AgentCapability) -> List[AgentInfo]:
        """Get all agents that have a specific capability"""
        return [agent for agent in self.agents.values() if capability in agent.capabilities]
    
    def get_agent(self, name: str) -> Optional[AgentInfo]:
        """Get agent by name"""
        return self.agents.get(name)

class DocumentOrchestrator:
    """Orchestrator agent that dynamically plans and executes document parsing strategies"""
    
    def __init__(self, registry: AgentRegistry, logger: JsonLogger):
        self.registry = registry
        self.logger = logger
        self.execution_history: List[TaskResult] = []
        
        # The main orchestrator agent
        self.orchestrator_agent = Agent(
            name="DocumentParsingOrchestrator",
            instructions="""You are the master orchestrator for document parsing and page indexing.
            
            Your role is to:
            1. Analyze document characteristics and requirements
            2. Choose optimal strategies based on available agents and document context
            3. Adapt strategies based on intermediate results
            4. Coordinate agent execution and handle failures
            5. Ensure high-quality final output
            
            Available agent capabilities:
            - TOC_DETECTION: Detect if document has table of contents
            - TOC_EXTRACTION: Extract TOC content from detected TOC
            - PAGE_INDEX_DETECTION: Check if TOC contains page numbers
            - JSON_TRANSFORMATION: Convert raw TOC to structured format
            - CONTENT_TOC_GENERATION: Generate TOC from document content analysis
            - PAGE_NUMBER_MAPPING: Map section titles to physical pages
            - TITLE_VERIFICATION: Verify title locations
            - ITEM_FIXING: Fix individual incorrect mappings
            - SUMMARY_GENERATION: Generate section summaries
            - DESCRIPTION_GENERATION: Generate document descriptions
            
            For each decision, consider:
            - Document complexity and size
            - Available computational resources
            - Required accuracy vs speed tradeoffs
            - Fallback strategies for failure cases
            - Agent reliability scores and capabilities
            
            Always respond in JSON format with your reasoning and chosen strategy.
            """,
            model="gpt-4.1"  # Use more powerful model for orchestration
        )
    
    async def analyze_document(self, page_list: List, context: DocumentContext) -> DocumentContext:
        """Analyze document characteristics to inform strategy selection"""
        
        analysis_prompt = f"""
        Analyze this document for parsing strategy selection:
        
        Basic Stats:
        - Total pages: {context.total_pages}
        - Total tokens: {context.total_tokens}
        - Average tokens per page: {context.avg_tokens_per_page}
        
        Sample pages (first 3):
        {json.dumps([page[0][:500] for page in page_list[:3]], indent=2)}
        
        Determine:
        1. Document type (academic, technical, legal, report, etc.)
        2. Structural complexity (simple, moderate, complex)
        3. Language and formatting style
        4. Estimated number of main sections
        5. Presence of special elements (figures, tables, appendices)
        
        Return JSON: {{
            "document_type": "string",
            "complexity": "simple|moderate|complex", 
            "language": "string",
            "estimated_sections": number,
            "special_elements": ["list"],
            "formatting_style": "string",
            "confidence": float
        }}
        """
        
        result = await run_specific_agent(self.orchestrator_agent, analysis_prompt)
        analysis = extract_json(result)
        
        # Update context with analysis results
        if analysis:
            context.document_type = analysis.get("document_type", "unknown")
            context.language = analysis.get("language", "unknown")
            context.estimated_sections = analysis.get("estimated_sections", 0)
            context.has_complex_structure = analysis.get("complexity") == "complex"
        
        return context
    
    async def plan_strategy(self, context: DocumentContext, requirements: Dict[str, Any]) -> StrategyPlan:
        """Plan the optimal parsing strategy based on document analysis"""
        
        planning_prompt = f"""
        Plan the optimal document parsing strategy:
        
        Document Context:
        {json.dumps(asdict(context), indent=2)}
        
        Requirements:
        {json.dumps(requirements, indent=2)}
        
        Available Agents and Capabilities:
        {json.dumps({name: {"capabilities": [c.value for c in info.capabilities], 
                            "reliability": info.reliability_score,
                            "complexity_level": info.complexity_level,
                            "description": info.description} 
                    for name, info in self.registry.agents.items()}, indent=2)}
        
        Consider:
        1. What's the best approach for this document type and complexity?
        2. Which agents should be used in what sequence?
        3. What are the main risks and how to mitigate them?
        4. What fallback strategies should be prepared?
        5. How to balance accuracy vs computational cost?
        
        Plan a comprehensive strategy with:
        - Primary strategy with agent sequence
        - 2-3 fallback strategies
        - Quality checkpoints and adaptation triggers
        - Estimated confidence and resource usage
        
        Return JSON: {{
            "primary_strategy": {{
                "name": "string",
                "agent_sequence": ["agent_names"],
                "reasoning": "detailed explanation",
                "estimated_confidence": float,
                "estimated_tokens": number,
                "quality_checkpoints": ["checkpoint descriptions"]
            }},
            "fallback_strategies": [
                {{
                    "name": "string", 
                    "trigger_condition": "when to use this fallback",
                    "agent_sequence": ["agent_names"],
                    "reasoning": "why this fallback"
                }}
            ],
            "adaptation_plan": {{
                "decision_points": ["when to adapt"],
                "adaptation_triggers": ["what conditions trigger changes"],
                "monitoring_metrics": ["what to track"]
            }}
        }}
        """
        
        result = await run_specific_agent(self.orchestrator_agent, planning_prompt)
        plan_data = extract_json(result)
        
        if not plan_data or "primary_strategy" not in plan_data:
            # Fallback to default strategy
            return self._create_default_strategy()
        
        primary = plan_data["primary_strategy"]
        fallbacks = [fb["name"] for fb in plan_data.get("fallback_strategies", [])]
        
        return StrategyPlan(
            strategy_name=primary["name"],
            agent_sequence=primary["agent_sequence"],
            fallback_strategies=fallbacks,
            estimated_confidence=primary.get("estimated_confidence", 0.7),
            estimated_tokens=primary.get("estimated_tokens", 10000),
            reasoning=primary.get("reasoning", "Default reasoning")
        )
    
    async def execute_strategy(self, strategy: StrategyPlan, page_list: List, context: DocumentContext) -> Dict[str, Any]:
        """Execute the planned strategy with adaptive decision making"""
        
        results = {}
        current_data = {"pages": page_list, "context": context}
        
        self.logger.info(f"Executing strategy: {strategy.strategy_name}")
        self.logger.info(f"Agent sequence: {strategy.agent_sequence}")
        
        for i, agent_name in enumerate(strategy.agent_sequence):
            agent_info = self.registry.get_agent(agent_name)
            if not agent_info:
                self.logger.error(f"Agent {agent_name} not found in registry")
                continue
            
            # Prepare task-specific prompt
            task_prompt = await self._prepare_agent_prompt(agent_info, current_data, i, strategy)
            
            try:
                # Execute agent
                result = await run_specific_agent(agent_info.agent, task_prompt)
                
                # Process result
                task_result = self._process_agent_result(result, agent_info, current_data)
                results[agent_name] = task_result
                
                # Update current data for next agent
                current_data = self._update_data_context(current_data, task_result)
                
                # Check if we need to adapt strategy
                should_adapt, adaptation = await self._should_adapt_strategy(
                    task_result, results, strategy, i
                )
                
                if should_adapt:
                    self.logger.info(f"Adapting strategy: {adaptation}")
                    # Implement strategy adaptation logic
                    new_strategy = await self._adapt_strategy(strategy, adaptation, results)
                    return await self.execute_strategy(new_strategy, page_list, context)
                
            except Exception as e:
                self.logger.error(f"Agent {agent_name} failed: {str(e)}")
                # Try fallback or continue with next agent
                continue
        
        return self._compile_final_results(results, current_data)
    
    async def _prepare_agent_prompt(self, agent_info: AgentInfo, data: Dict, step: int, strategy: StrategyPlan) -> str:
        """Prepare context-aware prompt for each agent"""
        
        # This would be more sophisticated in practice, tailored to each agent's specific needs
        base_context = f"""
        You are executing step {step + 1} of the "{strategy.strategy_name}" strategy.
        
        Previous results summary: {json.dumps({k: "completed" for k in data.keys() if k != "pages"}, indent=2)}
        
        Your specific task based on your capabilities: {[c.value for c in agent_info.capabilities]}
        """
        
        # Add agent-specific context based on capabilities
        if AgentCapability.TOC_DETECTION in agent_info.capabilities:
            return f"{base_context}\n\nAnalyze the following pages for table of contents:\n{data.get('sample_pages', '')}"
        elif AgentCapability.CONTENT_TOC_GENERATION in agent_info.capabilities:
            return f"{base_context}\n\nGenerate TOC from content:\n{data.get('content_pages', '')}"
        # Add more capability-specific prompts
        
        return base_context
    
    def _process_agent_result(self, result: str, agent_info: AgentInfo, context: Dict) -> TaskResult:
        """Process and validate agent results"""
        
        try:
            data = extract_json(result) if result.strip().startswith('{') else result
            return TaskResult(
                success=True,
                data=data,
                confidence=0.8,  # Would be calculated based on result quality
                tokens_used=count_tokens(result, "gpt-4.1-mini"),
                execution_time=0.0,  # Would be measured
                errors=[],
                agent_used=agent_info.name
            )
        except Exception as e:
            return TaskResult(
                success=False,
                data=None,
                confidence=0.0,
                tokens_used=0,
                execution_time=0.0,
                errors=[str(e)],
                agent_used=agent_info.name
            )
    
    def _update_data_context(self, current_data: Dict, task_result: TaskResult) -> Dict:
        """Update the data context for the next agent"""
        if task_result.success and task_result.data:
            current_data[f"result_{task_result.agent_used}"] = task_result.data
        return current_data
    
    async def _should_adapt_strategy(self, task_result: TaskResult, all_results: Dict, 
                                   strategy: StrategyPlan, step: int) -> Tuple[bool, str]:
        """Decide if strategy adaptation is needed"""
        
        # Simple adaptation logic - could be much more sophisticated
        if not task_result.success:
            return True, f"Agent failed at step {step}, need fallback"
        
        if task_result.confidence < 0.5:
            return True, f"Low confidence result at step {step}, consider alternative approach"
        
        return False, ""
    
    async def _adapt_strategy(self, current_strategy: StrategyPlan, reason: str, 
                            results: Dict) -> StrategyPlan:
        """Adapt the current strategy based on results"""
        
        adaptation_prompt = f"""
        Current strategy "{current_strategy.strategy_name}" needs adaptation.
        
        Reason: {reason}
        Current results: {json.dumps({k: v.success for k, v in results.items()}, indent=2)}
        
        Suggest an adapted strategy or select from fallbacks: {current_strategy.fallback_strategies}
        
        Return JSON with new strategy plan.
        """
        
        result = await run_specific_agent(self.orchestrator_agent, adaptation_prompt)
        adapted_plan = extract_json(result)
        
        # Return adapted strategy or fallback to default
        return self._create_strategy_from_plan(adapted_plan) if adapted_plan else self._create_default_strategy()
    
    def _create_default_strategy(self) -> StrategyPlan:
        """Create a default fallback strategy"""
        return StrategyPlan(
            strategy_name="conservative_fallback",
            agent_sequence=["toc_detector", "content_toc_generator", "page_mapper", "title_verifier"],
            fallback_strategies=["simple_content_analysis"],
            estimated_confidence=0.6,
            estimated_tokens=15000,
            reasoning="Default conservative approach when planning fails"
        )
    
    def _create_strategy_from_plan(self, plan: Dict) -> StrategyPlan:
        """Create StrategyPlan from orchestrator's plan"""
        return StrategyPlan(
            strategy_name=plan.get("name", "adapted_strategy"),
            agent_sequence=plan.get("agent_sequence", []),
            fallback_strategies=plan.get("fallback_strategies", []),
            estimated_confidence=plan.get("estimated_confidence", 0.5),
            estimated_tokens=plan.get("estimated_tokens", 10000),
            reasoning=plan.get("reasoning", "Adapted strategy")
        )
    
    def _compile_final_results(self, results: Dict[str, TaskResult], data: Dict) -> Dict[str, Any]:
        """Compile final results from all agent executions"""
        
        # Extract the final document structure from results
        final_structure = None
        for result in results.values():
            if result.success and isinstance(result.data, (list, dict)):
                final_structure = result.data
                break
        
        return {
            "success": any(r.success for r in results.values()),
            "structure": final_structure,
            "execution_summary": {
                "agents_used": list(results.keys()),
                "total_tokens": sum(r.tokens_used for r in results.values()),
                "overall_confidence": sum(r.confidence for r in results.values()) / len(results) if results else 0,
                "errors": [error for r in results.values() for error in r.errors]
            }
        }

# Main orchestrated parsing function
async def orchestrated_page_index(doc, requirements: Dict[str, Any] = None):
    """Main function using orchestrator-based approach"""
    
    if requirements is None:
        requirements = {
            "accuracy_threshold": 0.8,
            "max_tokens": 50000,
            "include_summaries": False,
            "include_descriptions": False,
            "timeout_minutes": 10
        }
    
    # Initialize components
    logger = JsonLogger(doc)
    registry = AgentRegistry()
    orchestrator = DocumentOrchestrator(registry, logger)
    
    # Get basic document info
    from pageindex.utils import get_page_tokens, get_pdf_name
    page_list = get_page_tokens(doc)
    
    # Create document context
    context = DocumentContext(
        total_pages=len(page_list),
        total_tokens=sum(page[1] for page in page_list),
        avg_tokens_per_page=sum(page[1] for page in page_list) / len(page_list),
        has_complex_structure=False,  # Will be determined by analysis
        estimated_sections=0  # Will be determined by analysis
    )
    
    logger.info(f"Starting orchestrated parsing for {context.total_pages} pages")
    
    try:
        # Analyze document
        context = await orchestrator.analyze_document(page_list, context)
        logger.info(f"Document analysis complete: {asdict(context)}")
        
        # Plan strategy
        strategy = await orchestrator.plan_strategy(context, requirements)
        logger.info(f"Strategy planned: {asdict(strategy)}")
        
        # Execute strategy
        results = await orchestrator.execute_strategy(strategy, page_list, context)
        logger.info(f"Strategy execution complete")
        
        return {
            "doc_name": get_pdf_name(doc),
            "structure": results.get("structure"),
            "execution_summary": results.get("execution_summary"),
            "document_context": asdict(context),
            "strategy_used": asdict(strategy)
        }
        
    except Exception as e:
        logger.error(f"Orchestrated parsing failed: {str(e)}")
        raise