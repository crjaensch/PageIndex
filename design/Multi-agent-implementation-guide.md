# Multi-Agent Orchestrator Implementation Guide

## Overview

This implementation creates a sophisticated multi-agent system that uses your existing 13 agents with intelligent orchestration and just-in-time data preparation through the `EnhancedPageIndexProcessingContext`.

## Key Architectural Components

### 1. **EnhancedPageIndexProcessingContext**
```python
class EnhancedPageIndexProcessingContext:
    # Just-in-time data preparation methods
    def get_grouped_content_for_toc_generation(self) -> List[str]
    def get_labeled_pages_for_mapping(self) -> str
    def get_verification_data_for_title(self, title: str, page: int) -> Dict
    
    # Agent result storage and retrieval
    def store_agent_result(self, agent_name: str, result: Any)
    def get_agent_result(self, agent_name: str) -> Any
    
    # Context updates from agent results
    def update_from_toc_detection(self, result: Dict)
    def update_from_page_mapping(self, result: List[Dict])
```

**Benefits:**
- **Lazy Loading**: Data prepared only when agents need it
- **Memory Efficient**: No unnecessary data duplication
- **Agent Communication**: Results from one agent available to others
- **State Tracking**: Complete processing history maintained

### 2. **SpecializedAgent Wrappers**
```python
class SpecializedAgent:
    async def execute(self, context: EnhancedPageIndexProcessingContext, 
                     task: AgentTask) -> AgentResult
    
    def _prepare_agent_prompt(self, context, task) -> str:
        # Agent-specific data preparation
        if self.name == "CREATE_TOC_FROM_CONTENT":
            grouped_content = context.get_grouped_content_for_toc_generation()
            # Return exact format as original process_no_toc
```

**Benefits:**
- **Consistent Interface**: All agents use same execution pattern
- **Context-Aware**: Each agent gets exactly the data it needs
- **Original Logic Preserved**: Same data formatting as your working functions
- **Error Handling**: Standardized error reporting and recovery

### 3. **MultiAgentOrchestrator**
```python
class MultiAgentOrchestrator:
    async def orchestrate_processing(self) -> Dict[str, Any]:
        while not_complete:
            decision = await self._get_orchestrator_decision()
            agent_result = await self.agents[decision.next_agent].execute(context, task)
            await self._update_context_from_agent_result(agent_result)
```

**Benefits:**
- **Intelligent Routing**: Orchestrator decides which agent to run next
- **Adaptive Strategy**: Can change approach based on intermediate results
- **Quality Monitoring**: Tracks success/failure and adapts accordingly
- **Resource Awareness**: Considers computational costs and constraints

## Information Flow Between Agents

### **Scenario 1: Document with No TOC**

```python
# 1. TOC_DETECTOR determines no TOC exists
toc_result = await toc_detector.execute(context, detection_task)
context.update_from_toc_detection({"has_toc": False})

# 2. Orchestrator decides to use content analysis
decision = orchestrator.decide_next_agent(context)
# Returns: {"next_agent": "CREATE_TOC_FROM_CONTENT"}

# 3. CREATE_TOC_FROM_CONTENT requests grouped content
grouped_content = context.get_grouped_content_for_toc_generation()
# Returns: ["<physical_index_1>\nContent...\n<physical_index_1>", ...]

# 4. Results stored for next agent
toc_content_result = await create_toc_agent.execute(context, content_task)
context.store_agent_result("CREATE_TOC_FROM_CONTENT", toc_content_result.data)

# 5. ADD_PAGE_NUMBER can now access previous results
previous_toc = context.get_agent_result("CREATE_TOC_FROM_CONTENT")
labeled_pages = context.get_labeled_pages_for_mapping()
```

### **Scenario 2: Document with TOC but No Page Numbers**

```python
# 1. TOC_DETECTOR finds TOC
context.update_from_toc_detection({"has_toc": True, "toc_pages": [2, 3]})

# 2. EXTRACT_TOC gets specific pages
toc_content = context.get_content_for_toc_extraction([2, 3])

# 3. TOC_JSON_TRANSFORMER structures the content
structured_toc = await transformer.execute(context, transform_task)
context.update_from_toc_transformation(structured_toc.data)

# 4. ADD_PAGE_NUMBER maps to physical pages
mapping_result = await mapper.execute(context, mapping_task)
context.update_from_page_mapping(mapping_result.data)
```

## Just-In-Time Data Preparation Examples

### **Content TOC Generation**
```python
def get_grouped_content_for_toc_generation(self, max_tokens: int = 20000) -> List[str]:
    if self._grouped_content_cache is None:
        page_contents = []
        token_lengths = []
        
        # EXACT same formatting as original process_no_toc
        for page_idx in range(1, len(self.page_list) + 1):
            page_text = f"<physical_index_{page_idx}>\n{self.page_list[page_idx-1][0]}\n<physical_index_{page_idx}>\n\n"
            page_contents.append(page_text)
            token_lengths.append(count_tokens(page_text, self.model))
        
        # Use original grouping logic
        self._grouped_content_cache = page_list_to_group_text(page_contents, token_lengths, max_tokens)
    
    return self._grouped_content_cache
```

**Key Features:**
- **Caching**: Expensive operations cached for reuse
- **Original Logic**: Identical to your working `process_no_toc` function
- **Flexible**: Can adjust `max_tokens` based on agent needs

### **Title Verification Data**
```python
def get_verification_data_for_title(self, title: str, physical_index: int) -> Dict[str, Any]:
    if 1 <= physical_index <= len(self.page_list):
        page_content = self.page_list[physical_index - 1][0]
        return {
            "title": title,
            "physical_index": physical_index,
            "page_content": page_content
        }
    return {"title": title, "physical_index": physical_index, "page_content": ""}
```

**Benefits:**
- **On-Demand**: Only prepares data when verification agent needs it
- **Precise**: Provides exact page content for fuzzy matching
- **Safe**: Handles edge cases (invalid page numbers)

## Orchestrator Decision Making

### **Context-Aware Decisions**
```python
async def _get_orchestrator_decision(self) -> Dict[str, Any]:
    context_summary = self.context.get_current_state_summary()
    # Includes: document characteristics, TOC status, processing progress
    
    decision_prompt = f"""
    Current processing state:
    {json.dumps(context_summary, indent=2)}
    
    Agent execution history:
    {self.context.processing_metadata["agents_used"]}
    
    Determine the next action based on current state and progress.
    """
    
    response = await run_specific_agent(self.orchestrator_agent, decision_prompt)
    return extract_json(response)
```

**Decision Factors:**
- **Document Type**: Academic papers vs technical manuals need different approaches
- **Processing History**: What agents have already run successfully
- **Quality Metrics**: Current accuracy and confidence levels
- **Resource Constraints**: Token usage and time limits

### **Adaptive Strategy Example**
```python
# If TOC extraction produces low-quality results
if toc_quality_score < 0.6:
    decision = {
        "next_agent": "CREATE_TOC_FROM_CONTENT",
        "reasoning": "TOC extraction quality too low, switching to content analysis",
        "completion_status": "adapt_strategy"
    }
```

## Integration with Original System

### **Drop-in Replacement**
```python
# Original function call
result = await page_index_main(doc, opt)

# Multi-agent equivalent
result = await multi_agent_page_index(doc, opt)
```

### **Backward Compatibility**
```python
# Your existing agents work unchanged
CHECK_TITLE_APPEARANCE_AGENT = Agent(name="...", instructions="...", model="...")

# Wrapped in specialized agent interface
specialized_agent = SpecializedAgent("CHECK_TITLE_APPEARANCE", 
                                   CHECK_TITLE_APPEARANCE_AGENT,
                                   "Check title appearances")
```

### **Gradual Migration**
```python
class HybridProcessor:
    def __init__(self, use_orchestrator_percentage: float = 0.1):
        self.orchestrator_usage = use_orchestrator_percentage
    
    async def process_document(self, doc, opt):
        if should_use_orchestrator(doc):
            return await multi_agent_page_index(doc, opt)
        else:
            return await page_index_main(doc, opt)  # Original
```

## Performance and Quality Benefits

### **Quality Improvements**
- **Adaptive Processing**: System learns which approaches work best
- **Error Recovery**: Failed agents trigger intelligent fallbacks
- **Quality Monitoring**: Continuous assessment and adaptation

### **Performance Optimizations**
- **Lazy Loading**: Data prepared only when needed
- **Caching**: Expensive operations cached across agents
- **Parallel Opportunities**: Orchestrator can identify parallelizable tasks

### **Maintainability Gains**
- **Modular Design**: Easy to add/modify individual agents
- **Clear Interfaces**: Standardized agent communication patterns
- **Rich Debugging**: Complete execution history and state tracking

## Testing Strategy

### **Unit Tests**
```python
class TestEnhancedContext:
    async def test_just_in_time_data_preparation(self):
        context = EnhancedPageIndexProcessingContext(test_doc)
        grouped_content = context.get_grouped_content_for_toc_generation()
        assert len(grouped_content) > 0
        assert "<physical_index_1>" in grouped_content[0]

class TestSpecializedAgent:
    async def test_agent_execution(self):
        agent = SpecializedAgent("TOC_DETECTOR", TOC_DETECTOR_AGENT, "...")
        result = await agent.execute(context, task)
        assert result.success
        assert result.data is not None
```

### **Integration Tests**
```python
class TestMultiAgentOrchestrator:
    async def test_full_processing_pipeline(self):
        result = await multi_agent_page_index(test_doc)
        assert result["structure"] is not None
        assert "processing_metadata" in result
```

## Migration Roadmap

### **Phase 1: Core Infrastructure (Week 1-2)**
- Implement `EnhancedPageIndexProcessingContext`
- Create `SpecializedAgent` wrappers
- Add just-in-time data preparation methods

### **Phase 2: Basic Orchestrator (Week 3-4)**
- Implement `MultiAgentOrchestrator`
- Add simple decision logic
- Test with subset of agents

### **Phase 3: Advanced Features (Week 5-6)**
- Add adaptive strategy logic
- Implement quality monitoring
- Add comprehensive error handling

### **Phase 4: Production Deployment (Week 7-8)**
- Performance optimization
- Comprehensive testing
- Gradual rollout with monitoring

This multi-agent architecture transforms your document parsing system into an intelligent, adaptive, and maintainable solution while preserving all your proven agent logic and data preparation methods.