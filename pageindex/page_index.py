import os
import math
import json
import copy
import random
import re
import asyncio
from io import BytesIO  # Add BytesIO import to fix the NameError

from agents import Agent
from .utils import run_specific_agent, extract_json, count_tokens, write_node_id, JsonLogger, add_preface_if_needed, get_page_tokens, post_processing, convert_physical_index_to_int, add_node_text, get_pdf_name, convert_page_to_int, generate_summaries_for_structure, generate_doc_description, remove_structure_text, ConfigLoader

# --- Agent Definitions ---
DEFAULT_AGENT_MODEL = "gpt-4.1-mini" # Or your preferred model, e.g., from config

CHECK_TITLE_APPEARANCE_AGENT = Agent(
    name="TitleAppearanceCheckerAgent",
    instructions="Your job is to check if a given section title appears or starts in the given page_text. Use fuzzy matching and ignore space inconsistencies. You must reply in the specified JSON format: {\"thinking\": \"<your reasoning>\", \"answer\": \"yes or no\"}. Directly return the final JSON structure. Do not output anything else.",
    model=DEFAULT_AGENT_MODEL
)

CHECK_TITLE_START_AGENT = Agent(
    name="TitleStartCheckerAgent",
    instructions="Your job is to check if the current section starts in the beginning of the given page_text. If there are other contents before the current section title, then it does not start at the beginning. Use fuzzy matching and ignore space inconsistencies. You must reply in the specified JSON format: {\\\"thinking\\\": \\\"<your reasoning>\\\", \\\"start_begin\\\": \\\"yes or no\\\"}. Directly return the final JSON structure. Do not output anything else.",
    model=DEFAULT_AGENT_MODEL
)

TOC_DETECTOR_AGENT = Agent(
    name="TocDetectorAgent",
    instructions="Your job is to detect if there is a table of content provided in the given text. Note that abstract, summary, notation list, figure list, table list, etc., are not table of contents. You must reply in the specified JSON format: {\\\"thinking\\\": \\\"<why you think there is a ToC>\\\", \\\"toc_detected\\\": \\\"yes or no\\\"}. Directly return the final JSON structure. Do not output anything else.",
    model=DEFAULT_AGENT_MODEL
)

TOC_EXTRACTION_COMPLETE_AGENT = Agent(
    name="TocExtractionCompleteAgent",
    instructions="Your job is to check if the given table of contents (ToC) is complete based on the provided text. The ToC might be a continuation of a previous extraction. You must reply in the specified JSON format: {\\\"thinking\\\": \\\"<your reasoning>\\\", \\\"complete\\\": \\\"yes or no\\\"}. Directly return the final JSON structure. Do not output anything else.",
    model=DEFAULT_AGENT_MODEL
)

EXTRACT_TOC_AGENT = Agent(
    name="TocExtractorAgent",
    instructions="Your job is to extract the full table of contents (ToC) from the given text. Replace '...' with ':' if appropriate. Ensure all relevant sections are included. Directly return the full table of contents as a single block of text. Do not output anything else.",
    model=DEFAULT_AGENT_MODEL
)

TOC_JSON_TRANSFORMER_AGENT = Agent(
    name="TocJsonTransformerAgent",
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
    model=DEFAULT_AGENT_MODEL
)

CREATE_TOC_FROM_CONTENT_AGENT = Agent(
    name="CreateTocFromContentAgent",
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
    model=DEFAULT_AGENT_MODEL
)

SINGLE_TOC_ITEM_FIXER_AGENT = Agent(
    name="SingleTocItemFixerAgent",
    instructions="""You are an expert in finding the physical page number for a single table of contents (ToC) item title within a given text snippet.
Given a "Section Title" and "Partial Document Text" (which includes <physical_index_X> tags):
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
    model=DEFAULT_AGENT_MODEL  # Or your preferred default model
)

TOC_TRANSFORMATION_COMPLETE_AGENT = Agent(
    name="TocTransformationCompleteAgent",
    instructions="""You are given a raw table of contents (ToC) text and a cleaned/structured ToC (usually in JSON format).
Your job is to determine if the cleaned ToC accurately and completely represents all the information from the raw ToC.
Consider if all sections, page numbers (if present in raw), and hierarchical relationships from the raw ToC are present in the cleaned ToC.

Reply in a JSON format:
{
    "thinking": "<Explain your reasoning why the cleaned table of contents is complete or not, comparing it to the raw version.>",
    "completed": "<'yes' or 'no'>"
}
Directly return the final JSON structure. Do not output anything else.
""",
    model=DEFAULT_AGENT_MODEL
)

ADD_PAGE_NUMBER_AGENT = Agent(
    name="AddPageNumberAgent",
    instructions="""You are an expert in correlating table of contents (ToC) items with their physical page numbers in a document.
You will be given:
1. A 'Partial Document Text' which may contain tags like <physical_index_X> (e.g., <physical_index_123>) indicating page numbers.
2. A 'Given ToC Structure' as a JSON list of objects, where each object has "structure" and "title" keys (and potentially an existing "physical_index" which should be preserved if the item is not found in the current 'Partial Document Text').

Your task is to process each item in the 'Given ToC Structure'. For each item:
- Determine if its 'title' starts or appears in the 'Partial Document Text'.
- If it does, find the *first* <physical_index_X> tag that appears at or after the title's occurrence in the text. This tag represents the page number for this occurrence.
- Construct an output JSON list. Each item in this output list should correspond to an item in the input 'Given ToC Structure'.
- The output item must have the following keys:
    - "structure": (string or null) The original structure index from the input.
    - "title": (string) The original title from the input.
    - "physical_index": (string or null) The extracted <physical_index_X> tag (e.g., "<physical_index_123>") if found for the title in the current 'Partial Document Text'. If the title is not found, or no <physical_index_X> tag is associated with its occurrence in this specific part, the "physical_index" should retain its original value from the input 'Given ToC Structure' (if one existed) or be null if none existed and none was found.

Important:
- Only consider the 'Partial Document Text' provided for finding titles and associating new page numbers.
- The output must be a valid JSON list of objects.
Directly return *only* the final JSON list. Do not add any explanatory text before or after the JSON.
""",
    model=DEFAULT_AGENT_MODEL
)

PAGE_INDEX_DETECTOR_AGENT = Agent(
    name="PageIndexDetectorAgent",
    instructions="""You will be given a table of contents (ToC) text.
Your job is to detect if there are page numbers or page indices explicitly mentioned within this ToC text.
For example, entries like "Chapter 1 ..... 5" or "Section A ... Page 12" indicate page numbers are present.
If entries are just titles like "Introduction", "Conclusion" without any trailing numbers that look like page references, then page numbers are not given.

Reply in a JSON format:
{
    "thinking": "<Explain your reasoning why you think page numbers/indices are present or absent in the given ToC text.>",
    "page_index_given_in_toc": "<'yes' or 'no'>"
}
Directly return the final JSON structure. Do not output anything else.
""",
    model=DEFAULT_AGENT_MODEL
)
# --- End Agent Definitions ---


################### check title in page #########################################################
async def check_title_appearance(item, page_list, start_index=1, model=None):
    title = item['title']
    if 'physical_index' not in item or item['physical_index'] is None:
        return {'list_index': item.get('list_index'), 'answer': 'no', 'title': title, 'page_number': None}

    page_number = item['physical_index']
    page_text = page_list[page_number - start_index][0]

    user_prompt_content = f"""
    Please check the following:
    Section Title: {title}
    Page Text:
    ---
    {page_text}
    ---
    Remember to reply in the JSON format as specified in your instructions.
    """ # The JSON structure is now part of the agent's instructions

    # Pass model_override parameter to allow using a more powerful model when needed
    raw_response = await run_specific_agent(CHECK_TITLE_APPEARANCE_AGENT, user_prompt_content, model_override=model)
    response_json = extract_json(raw_response)

    answer = 'no' # Default answer
    if isinstance(response_json, dict) and 'answer' in response_json:
        answer = response_json['answer']
    # Consider logging if raw_response indicates an error from run_specific_agen
    elif "Error: Agent execution failed" in str(raw_response):
        # Optionally log the error from raw_response if needed, e.g., using a logger object if available
        # print(f"Agent execution failed for title '{title}': {raw_response}") # Or use proper logging
        pass # Defaults to 'no'

    return {'list_index': item['list_index'], 'answer': answer, 'title': title, 'page_number': page_number}


async def check_title_appearance_in_start(title, page_text, logger=None): # model parameter removed
    user_prompt_content = f"""
    Please check if the section starts at the beginning of the page text:
    Section Title: {title}
    Page Text:
    ---
    {page_text}
    ---
    Remember to reply in the JSON format as specified in your instructions.
    """ # JSON structure is in agent instructions

    raw_response = await run_specific_agent(CHECK_TITLE_START_AGENT, user_prompt_content)
    response_json = extract_json(raw_response)

    if logger:
        # Log the structured JSON or raw response based on preference
        logger.info(f"Agent response for title '{title}' start check: {response_json if isinstance(response_json, dict) else raw_response}")

    start_begin = 'no' # Defaul
    if isinstance(response_json, dict) and 'start_begin' in response_json:
        start_begin = response_json['start_begin']
    elif "Error: Agent execution failed" in str(raw_response) and logger:
        logger.error(f"Agent execution failed for title '{title}' start check: {raw_response}")

    return start_begin


async def check_title_appearance_in_start_concurrent(structure, page_list, logger=None):
    if logger:
        logger.info("Checking title appearance in start concurrently")

    # skip items without physical_index
    for item in structure:
        if item.get('physical_index') is None:
            item['appear_start'] = 'no'

    # only for items with valid physical_index
    tasks = []
    valid_items = []
    for item in structure:
        if item.get('physical_index') is not None:
            page_text = page_list[item['physical_index'] - 1][0]
            tasks.append(check_title_appearance_in_start(item['title'], page_text, logger=logger))
            valid_items.append(item)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for item, result in zip(valid_items, results):
        if isinstance(result, Exception):
            if logger:
                logger.error(f"Error checking start for {item['title']}: {result}")
            item['appear_start'] = 'no'
        else:
            item['appear_start'] = result

    return structure


async def toc_detector_single_page(content, model=None):
    user_prompt_content = f"""
    Given text:
    ---
    {content}
    ---
    Please detect if a table of contents is present and reply in the JSON format as specified in your instructions.
    """

    # Pass model_override parameter to allow using a more powerful model when needed
    raw_response = await run_specific_agent(TOC_DETECTOR_AGENT, user_prompt_content, model_override=model)
    json_content = extract_json(raw_response)

    toc_detected = 'no' # Defaul
    if isinstance(json_content, dict) and 'toc_detected' in json_content:
        toc_detected = json_content['toc_detected']
    elif "Error: Agent execution failed" in str(raw_response):
        # Handle or log error, e.g.
        # print(f"Agent execution failed for toc_detector_single_page: {raw_response}")
        pass # Defaults to 'no'

    return toc_detected


async def check_if_toc_extraction_is_complete(content, toc): # Changed to async, model removed
    user_prompt_content = f"""
    Given text:
    ---
    {content}
    ---
    Given table of contents (potentially partial or continued):
    ---
    {toc}
    ---
    Is the table of contents extraction complete? Please reply in the JSON format specified in your instructions.
    """

    raw_response = await run_specific_agent(TOC_EXTRACTION_COMPLETE_AGENT, user_prompt_content)
    json_content = extract_json(raw_response)

    is_complete = 'no' # Defaul
    if isinstance(json_content, dict) and 'complete' in json_content:
        is_complete = json_content['complete']
    elif "Error: Agent execution failed" in str(raw_response):
        # Optionally log the error
        # print(f"Agent execution failed for check_if_toc_extraction_is_complete: {raw_response}")
        pass # Defaults to 'no'

    return is_complete


async def check_if_toc_transformation_is_complete(content, toc, logger=None): # model param removed, made async
    user_prompt_content = f"""
Raw Table of contents:
---
{content}
---
Cleaned/Structured Table of contents:
---
{toc}
---
Based on these, is the cleaned/structured ToC complete and accurate representation of the raw ToC?
Please reply in the JSON format specified in your instructions for TOC_TRANSFORMATION_COMPLETE_AGENT.
"""
    raw_response = await run_specific_agent(TOC_TRANSFORMATION_COMPLETE_AGENT, user_prompt_content)
    json_content = extract_json(raw_response)

    completed_status = 'no' # Defaul
    if isinstance(json_content, dict) and 'completed' in json_content:
        completed_status = json_content['completed']
    elif "Error: Agent execution failed" in str(raw_response):
        if logger:
            logger.error(f"Agent execution failed for check_if_toc_transformation_is_complete: {raw_response}")
        # Defaulting to 'no' as a safe measure

    if logger:
        logger.info(f"check_if_toc_transformation_is_complete response: {json_content if isinstance(json_content, dict) else raw_response}, status: {completed_status}")

    return completed_status

async def extract_toc_content(content, model=None):
    user_prompt_content = f"""
    Please extract the full table of contents from the following text:
    ---
    {content}
    ---
    Remember to replace '...' with ':' where appropriate and return only the complete table of contents text.
    """

    # The agent is expected to return the ToC text directly.
    # run_specific_agent returns the 'final_output' from the agent.
    # Pass model_override parameter to allow using a more powerful model for this complex task
    toc_text = await run_specific_agent(EXTRACT_TOC_AGENT, user_prompt_content, model_override=model)

    # Check if the agent call itself failed (e.g. API key issue, agent SDK error)
    if "Error: Agent execution failed" in str(toc_text):
        # Handle or log error, perhaps return an empty string or raise an exception
        # print(f"Agent execution failed for extract_toc_content: {toc_text}")
        # Consider using logger if available: logger.error(f"Agent execution failed for extract_toc_content: {toc_text}")
        return "" # Or raise an exception based on desired error handling

    return toc_text

async def detect_page_index(toc_content, logger=None): # model param removed, made async
    # print('start detect_page_index') # Optional: keep if debugging needed
    user_prompt_content = f"""
Given table of contents text:
---
{toc_content}
---
Are page numbers/indices present in this table of contents?
Please reply in the JSON format specified in your instructions for PAGE_INDEX_DETECTOR_AGENT.
"""
    raw_response = await run_specific_agent(PAGE_INDEX_DETECTOR_AGENT, user_prompt_content)
    json_content = extract_json(raw_response)

    page_index_given = 'no' # Default
    if isinstance(json_content, dict) and 'page_index_given_in_toc' in json_content:
        page_index_given = json_content['page_index_given_in_toc']
    elif "Error: Agent execution failed" in str(raw_response):
        if logger:
            logger.error(f"Agent execution failed for detect_page_index: {raw_response}")
        # Defaulting to 'no' as a safe measure

    if logger:
        logger.info(f"detect_page_index response: {json_content if isinstance(json_content, dict) else raw_response}, page_index_given: {page_index_given}")

    return page_index_given

async def toc_extractor(page_list, toc_page_list, logger=None): # Made async, model param removed, logger added
    def transform_dots_to_colon(text):
        # Replace 5 or more consecutive dots with ': '
        text = re.sub(r'\.{5,}', ': ', text)
        # Handle dots separated by spaces, e.g., ". . . . ."
        text = re.sub(r'(?:\. ){4,}\.', ': ', text) # Matches '. . . . .' (at least 5 dots with 4 spaces)
        return text

    toc_content_parts = []
    for page_idx in toc_page_list:
        # Ensure page_idx is within bounds of page_list
        if 0 <= page_idx < len(page_list):
            toc_content_parts.append(page_list[page_idx][0])
        elif logger:
            logger.warning(f"toc_extractor: page_idx {page_idx} out of bounds for page_list (len {len(page_list)}). Skipping.")

    toc_content = "".join(toc_content_parts)
    toc_content = transform_dots_to_colon(toc_content)

    # Pass logger to detect_page_index
    has_page_index = await detect_page_index(toc_content, logger=logger)

    if logger:
        logger.info(f"toc_extractor result: page_index_given_in_toc='{has_page_index}'")

    return {
        "toc_content": toc_content,
        "page_index_given_in_toc": has_page_index
    }

async def toc_index_extractor(toc, content): # Made async, model removed, uses ADD_PAGE_NUMBER_AGENT
    # print('start toc_index_extractor') # Optional: keep if debugging needed
    # Ensure toc is a JSON string for the agent. If it's already a Python list/dict, json.dumps it.
    # Assuming 'toc' is a Python object (list of dicts) as per typical usage.
    toc_json_string = json.dumps(toc, indent=2)

    user_prompt_content = f"""
    Partial Document Text:
    ---
    {content}
    ---
    Given ToC Structure:
    ---
    {toc_json_string}
    ---
    Please process the "Given ToC Structure" against the "Partial Document Text" and return an updated JSON list with "physical_index" fields populated or preserved as per your instructions for ADD_PAGE_NUMBER_AGENT.
    Focus on adding physical_index to sections found in the provided pages. If a section isn't in these pages, its physical_index should remain unchanged from the input ToC Structure.
    """

    raw_response = await run_specific_agent(ADD_PAGE_NUMBER_AGENT, user_prompt_content)
    json_result = extract_json(raw_response)

    if not isinstance(json_result, list):
        # Fallback or error handling if agent doesn't return a lis
        # print(f"toc_index_extractor: Agent returned malformed JSON or not a list. Raw: {raw_response}")
        # Consider using a logger and returning the original toc or an empty lis
        return toc # Or [] or raise Exception("Agent failed to return valid JSON list for toc_index_extractor")

    return json_result



async def toc_transformer(toc_content, model=None):
    print('start toc_transformer')
    user_prompt_content = f"""
    Please transform the following table of contents text into the JSON structure specified in your instructions.
    Raw Table of Contents Text:
    ---
    {toc_content}
    ---
    """

    # Pass model_override parameter to allow using a more powerful model for this complex JSON transformation task
    raw_response = await run_specific_agent(TOC_JSON_TRANSFORMER_AGENT, user_prompt_content, model_override=model)

    transformed_json = extract_json(raw_response) # extract_json should handle potential markdown/text around JSON

    if not isinstance(transformed_json, dict) or 'table_of_contents' not in transformed_json:
        # Log error or handle appropriately if agent fails or returns unexpected structure
        # For now, returning empty list to somewhat match original error paths
        # print(f"toc_transformer: Agent failed or returned malformed JSON. Raw: {raw_response}")
        # Consider using a logger if available: logger.error(f"toc_transformer: Agent failed or returned malformed JSON. Raw: {raw_response}")
        return []

    # convert_page_to_int is assumed to be synchronous and operate on the lis
    cleaned_toc_list = convert_page_to_int(transformed_json['table_of_contents'])

    return cleaned_toc_list


async def find_toc_pages(start_page_index, page_list, opt, logger=None, model=None):
    print('start find_toc_pages')
    last_page_is_yes = False
    toc_page_list = []
    i = start_page_index

    # Use the model from opt if provided and no specific model override is given
    effective_model = model if model is not None else getattr(opt, 'model', None)

    while i < len(page_list):
        # Only check beyond max_pages if we're still finding TOC pages
        if i >= opt.toc_check_page_num and not last_page_is_yes:
            break
        # Call the now async toc_detector_single_page with model parameter
        detected_result = await toc_detector_single_page(page_list[i][0], model=effective_model)
        if detected_result == 'yes':
            if logger:
                logger.info(f'Page {i} has toc')
            toc_page_list.append(i)
            last_page_is_yes = True
        elif detected_result == 'no' and last_page_is_yes: # This logic remains the same
            if logger:
                logger.info(f'Found the last page with toc: {i-1}')
            break
        i += 1

    if not toc_page_list and logger:
        logger.info('No toc found')

    return toc_page_list


def remove_page_number(data):
    if isinstance(data, dict):
        data.pop('page_number', None)
        for key in list(data.keys()):
            if 'nodes' in key:
                remove_page_number(data[key])
    elif isinstance(data, list):
        for item in data:
            remove_page_number(item)
    return data


def extract_matching_page_pairs(toc_page, toc_physical_index, start_page_index):
    pairs = []
    for phy_item in toc_physical_index:
        for page_item in toc_page:
            if phy_item.get('title') == page_item.get('title'):
                physical_index = phy_item.get('physical_index')
                if physical_index is not None and int(physical_index) >= start_page_index:
                    pairs.append({
                        'title': phy_item.get('title'),
                        'page': page_item.get('page'),
                        'physical_index': physical_index
                    })
    return pairs


def calculate_page_offset(pairs):
    differences = []
    for pair in pairs:
        try:
            physical_index = pair['physical_index']
            page_number = pair['page']
            difference = physical_index - page_number
            differences.append(difference)
        except (KeyError, TypeError):
            continue

    if not differences:
        return None

    difference_counts = {}
    for diff in differences:
        difference_counts[diff] = difference_counts.get(diff, 0) + 1

    most_common = max(difference_counts.items(), key=lambda x: x[1])[0]

    return most_common

def add_page_offset_to_toc_json(data, offset):
    for i in range(len(data)):
        if data[i].get('page') is not None and isinstance(data[i]['page'], int):
            data[i]['physical_index'] = data[i]['page'] + offset
            del data[i]['page']

    return data


def page_list_to_group_text(page_contents, token_lengths, max_tokens=20000, overlap_page=1):
    num_tokens = sum(token_lengths)

    if num_tokens <= max_tokens:
        # merge all pages into one tex
        page_text = "".join(page_contents)
        return [page_text]

    subsets = []
    current_subset = []
    current_token_count = 0

    expected_parts_num = math.ceil(num_tokens / max_tokens)
    average_tokens_per_part = math.ceil(((num_tokens / expected_parts_num) + max_tokens) / 2)

    for i, (page_content, page_tokens) in enumerate(zip(page_contents, token_lengths)):
        if current_token_count + page_tokens > average_tokens_per_part:

            subsets.append(''.join(current_subset))
            # Start new subset from overlap if specified
            overlap_start = max(i - overlap_page, 0)
            current_subset = page_contents[overlap_start:i]
            current_token_count = sum(token_lengths[overlap_start:i])

        # Add current page to the subset
        current_subset.append(page_content)
        current_token_count += page_tokens

    # Add the last subset if it contains any pages
    if current_subset:
        subsets.append(''.join(current_subset))

    print('divide page_list to groups', len(subsets))
    return subsets

async def add_page_number_to_toc(part, structure): # Changed to async, model removed
    # Convert structure (Python list of dicts) to JSON string for the agent promp
    structure_json_string = json.dumps(structure, indent=2)

    user_prompt_content = f"""
    Partial Document Text:
    ---
    {part}
    ---
    Given ToC Structure:
    ---
    {structure_json_string}
    ---
    Please process the "Given ToC Structure" against the "Partial Document Text" and return an updated JSON list with "physical_index" fields populated or preserved as per your instructions.
    """

    raw_response = await run_specific_agent(ADD_PAGE_NUMBER_AGENT, user_prompt_content)

    json_result = extract_json(raw_response)

    if not isinstance(json_result, list):
        # Handle error: agent did not return a list or JSON was malformed
        # print(f"add_page_number_to_toc: Agent returned malformed JSON or not a list. Raw: {raw_response}")
        # Consider using a logger if available
        # Fallback to returning the original structure or an empty list, or raise an error
        return structure # Or [] or raise Exception("Agent failed to return valid JSON list for add_page_number_to_toc")

    # The agent should directly return the desired structure, so no need to delete 'start' fields.
    return json_result


def remove_first_physical_index_section(text):
    """
    Removes the first section between <physical_index_X> and <physical_index_X> tags,
    and returns the remaining text.
    """
    pattern = r'<physical_index_\d+>.*?<physical_index_\d+>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        # Remove the first matched section
        return text.replace(match.group(0), '', 1)
    return text


async def process_no_toc(page_list, start_index=1, logger=None, model=None):
    page_contents=[]
    token_lengths=[]
    # Determine model for count_tokens, perhaps from DEFAULT_AGENT_MODEL or a config
    # For now, assuming count_tokens can handle model=None or we use a default like 'gpt-4o'
    # This might need adjustment if count_tokens strictly requires a model string.
    token_model_for_counting = DEFAULT_AGENT_MODEL # Or some other appropriate model string

    for page_idx in range(start_index, start_index+len(page_list)):
        page_text = f"<physical_index_{page_idx}>\n{page_list[page_idx-start_index][0]}\n<physical_index_{page_idx}>\n\n"
        page_contents.append(page_text)
        token_lengths.append(count_tokens(page_text, token_model_for_counting)) # Pass a model for token counting

    group_texts = page_list_to_group_text(page_contents, token_lengths)
    if logger:
        logger.info(f'process_no_toc: len(group_texts): {len(group_texts)}')

    full_toc_from_agent = []
    for i, group_text_chunk in enumerate(group_texts):
        user_prompt_content = f"""
        Document Text Chunk {i+1}/{len(group_texts)}:
        ---
        {group_text_chunk}
        ---
        Please generate table of contents items from this text chunk according to your instructions.
        If this is not the first chunk (i.e., chunk number > 1), assume it might be a continuation of a ToC from previous chunk(s) when determining structure numbers.
        """

        # print(f"Calling CREATE_TOC_FROM_CONTENT_AGENT for chunk {i+1}") # For debugging
        # Pass model_override parameter to allow using a more powerful model for this complex task
        raw_response = await run_specific_agent(CREATE_TOC_FROM_CONTENT_AGENT, user_prompt_content, model_override=model)
        chunk_toc_items = extract_json(raw_response)

        if isinstance(chunk_toc_items, list):
            full_toc_from_agent.extend(chunk_toc_items)
        else:
            if logger:
                logger.warning(f"process_no_toc: Agent returned non-list or malformed JSON for chunk {i+1}. Raw: {raw_response}. Skipping this chunk's ToC items.")
            # print(f"process_no_toc: Agent returned non-list for chunk {i+1}. Raw: {raw_response}") # For debugging

    if logger:
        logger.info(f'process_no_toc: Raw ToC from agent: {full_toc_from_agent}')

    # It's possible the agent might sometimes return numbers in physical_index, but prompt asks for string tag.
    # convert_physical_index_to_int expects string tags like "<physical_index_123>"
    toc_with_page_number_int = convert_physical_index_to_int(full_toc_from_agent)
    if logger:
        logger.info(f'process_no_toc: Converted physical_index to int: {toc_with_page_number_int}')

    return toc_with_page_number_int


async def process_toc_no_page_numbers(toc_content, toc_page_list, page_list,  start_index=1, logger=None): # Made async, model param removed
    page_contents=[]
    token_lengths=[]
    toc_content = await toc_transformer(toc_content) # Now async, model param removed
    logger.info(f'toc_transformer: {toc_content}')
    for page_idx in range(start_index, start_index+len(page_list)):
        page_text = f"<physical_index_{page_idx}>\n{page_list[page_idx-start_index][0]}\n<physical_index_{page_idx}>\n\n"
        page_contents.append(page_text)
        token_lengths.append(count_tokens(page_text, DEFAULT_AGENT_MODEL)) # Use default agent model for token counting

    group_texts = page_list_to_group_text(page_contents, token_lengths)
    logger.info(f'len(group_texts): {len(group_texts)}')

    toc_with_page_number=copy.deepcopy(toc_content)
    for group_text in group_texts:
        toc_with_page_number = await add_page_number_to_toc(group_text, toc_with_page_number) # Now async, model param removed
    logger.info(f'add_page_number_to_toc: {toc_with_page_number}')

    toc_with_page_number = convert_physical_index_to_int(toc_with_page_number)
    logger.info(f'convert_physical_index_to_int: {toc_with_page_number}')

    return toc_with_page_number



async def process_toc_with_page_numbers(toc_content, toc_page_list, page_list, toc_check_page_num=None, logger=None): # Made async, model param removed
    toc_with_page_number = await toc_transformer(toc_content) # Now async, model param removed
    logger.info(f'toc_with_page_number: {toc_with_page_number}')

    toc_no_page_number = remove_page_number(copy.deepcopy(toc_with_page_number))

    start_page_index = toc_page_list[-1] + 1
    main_content = ""
    for page_idx in range(start_page_index, min(start_page_index + toc_check_page_num, len(page_list))):
        main_content += f"<physical_index_{page_idx+1}>\n{page_list[page_idx][0]}\n<physical_index_{page_idx+1}>\n\n"

    toc_with_physical_index = await toc_index_extractor(toc_no_page_number, main_content) # Now async, model param removed
    logger.info(f'toc_with_physical_index: {toc_with_physical_index}')

    toc_with_physical_index = convert_physical_index_to_int(toc_with_physical_index)
    logger.info(f'toc_with_physical_index: {toc_with_physical_index}')

    matching_pairs = extract_matching_page_pairs(toc_with_page_number, toc_with_physical_index, start_page_index)
    logger.info(f'matching_pairs: {matching_pairs}')

    offset = calculate_page_offset(matching_pairs)
    logger.info(f'offset: {offset}')

    toc_with_page_number = add_page_offset_to_toc_json(toc_with_page_number, offset)
    logger.info(f'toc_with_page_number: {toc_with_page_number}')

    toc_with_page_number = await process_none_page_numbers(toc_with_page_number, page_list) # Now async, model param removed
    logger.info(f'toc_with_page_number: {toc_with_page_number}')

    return toc_with_page_number



##check if needed to process none page numbers
async def process_none_page_numbers(toc_items, page_list, start_index=1): # Made async, model param removed
    for i, item in enumerate(toc_items):
        if "physical_index" not in item:
            # logger.info(f"fix item: {item}")
            # Find previous physical_index
            prev_physical_index = 0  # Default if no previous item exists
            for j in range(i - 1, -1, -1):
                if toc_items[j].get('physical_index') is not None:
                    prev_physical_index = toc_items[j]['physical_index']
                    break

            # Find next physical_index
            next_physical_index = -1  # Default if no next item exists
            for j in range(i + 1, len(toc_items)):
                if toc_items[j].get('physical_index') is not None:
                    next_physical_index = toc_items[j]['physical_index']
                    break

            page_contents = []
            for page_idx in range(prev_physical_index, next_physical_index+1):
                # Add bounds checking to prevent IndexError
                list_index = page_idx - start_index
                if 0 <= list_index < len(page_list):
                    page_text = f"<physical_index_{page_idx}>\n{page_list[list_index][0]}\n<physical_index_{page_idx}>\n\n"
                    page_contents.append(page_text)
                else:
                    continue

            item_copy = copy.deepcopy(item)
            item_copy.pop('page', None)
            result = await add_page_number_to_toc(page_contents, item_copy) # Now async, model param removed
            if isinstance(result[0]['physical_index'], str) and result[0]['physical_index'].startswith('<physical_index'):
                item['physical_index'] = int(result[0]['physical_index'].split('_')[-1].rstrip('>').strip())
                item.pop('page', None)

    return toc_items


async def check_toc(page_list, opt=None, logger=None):
    toc_page_list = await find_toc_pages(start_page_index=0, page_list=page_list, opt=opt)
    if len(toc_page_list) == 0:
        if logger:
            logger.info('no toc found')
        else:
            print('no toc found')
        return await process_no_toc(page_list, start_index=1, logger=logger) # Now async, model param removed
    else:
        if logger:
            logger.info('toc found')
        else:
            print('toc found')

        toc_json = await toc_extractor(page_list, toc_page_list, logger=logger)

        if toc_json['page_index_given_in_toc'] == 'yes':
            if logger:
                logger.info('index found')
            else:
                print('index found')
            return {'toc_content': toc_json['toc_content'], 'toc_page_list': toc_page_list, 'page_index_given_in_toc': 'yes'}
        
        current_start_index = toc_page_list[-1] + 1
        
        # Loop to find additional TOC pages with page numbers
        while (current_start_index < len(page_list) and 
               current_start_index < opt.toc_check_page_num):
               
            additional_toc_pages = await find_toc_pages(
                start_page_index=current_start_index,
                page_list=page_list,
                opt=opt
            )

            if len(additional_toc_pages) == 0:
                break

            additional_toc_json = await toc_extractor(page_list, additional_toc_pages, logger=logger)
            if additional_toc_json['page_index_given_in_toc'] == 'yes':
                if logger:
                    logger.info('index found')
                else:
                    print('index found')
                return {'toc_content': additional_toc_json['toc_content'], 'toc_page_list': additional_toc_pages, 'page_index_given_in_toc': 'yes'}

            current_start_index = additional_toc_pages[-1] + 1
    
        if logger:
            logger.info('index not found')
        else:
            print('index not found')
        return {'toc_content': toc_json['toc_content'], 'toc_page_list': toc_page_list, 'page_index_given_in_toc': 'no'}



################### fix incorrect toc #########################################################
async def single_toc_item_index_fixer(section_title, content, logger=None): # Made async, model param removed
    user_prompt_content = f"""
Section Title:
---
{section_title}
---
Partial Document Text:
---
{content}
---
Please find the physical_index for the section title in the provided text, adhering to the SINGLE_TOC_ITEM_FIXER_AGENT instructions.
"""
    raw_response = await run_specific_agent(SINGLE_TOC_ITEM_FIXER_AGENT, user_prompt_content)
    json_content = extract_json(raw_response)

    if json_content and isinstance(json_content, dict) and 'physical_index' in json_content:
        physical_index_str = json_content['physical_index']
        if physical_index_str: # Check if it's not None or empty
            return convert_physical_index_to_int(physical_index_str)

    # Fallback if physical_index is not found, is None, or JSON is malformed
    # Depending on desired behavior, you might return None, raise an error, or return a specific value.
    # Returning None or a value that signifies "not found" is often safest.
    # The original function would error if 'physical_index' was missing or if convert_physical_index_to_int failed.
    # Let's return None to indicate it couldn't be determined, which calling functions might need to handle.
    if logger:
        logger.warning(f"single_toc_item_index_fixer could not determine physical_index for title '{section_title}'. Raw response: {raw_response}")
    return None # Or handle error as appropriate



async def fix_incorrect_toc(toc_with_page_number, page_list, incorrect_results, start_index=1, logger=None):
    if logger:
        logger.info(f'start fix_incorrect_toc with {len(incorrect_results)} incorrect results')
    else:
        print(f'start fix_incorrect_toc with {len(incorrect_results)} incorrect results')
    incorrect_indices = {result['list_index'] for result in incorrect_results}

    end_index = len(page_list) + start_index - 1

    incorrect_results_and_range_logs = []
    # Helper function to process and check a single incorrect item
    async def process_and_check_item(incorrect_item):
        list_index = incorrect_item['list_index']

        # Check if list_index is valid
        if list_index < 0 or list_index >= len(toc_with_page_number):
            # Return an invalid result for out-of-bounds indices
            return {
                'list_index': list_index,
                'title': incorrect_item['title'],
                'physical_index': incorrect_item.get('physical_index'),
                'is_valid': False
            }

        # Find the previous correct item
        prev_correct = None
        for i in range(list_index-1, -1, -1):
            if i not in incorrect_indices and 0 <= i < len(toc_with_page_number):
                physical_index = toc_with_page_number[i].get('physical_index')
                if physical_index is not None:
                    prev_correct = physical_index
                    break
        # If no previous correct item found, use start_index
        if prev_correct is None:
            prev_correct = start_index - 1

        # Find the next correct item
        next_correct = None
        for i in range(list_index+1, len(toc_with_page_number)):
            if i not in incorrect_indices and 0 <= i < len(toc_with_page_number):
                physical_index = toc_with_page_number[i].get('physical_index')
                if physical_index is not None:
                    next_correct = physical_index
                    break
        # If no next correct item found, use end_index
        if next_correct is None:
            next_correct = end_index

        incorrect_results_and_range_logs.append({
            'list_index': list_index,
            'title': incorrect_item['title'],
            'prev_correct': prev_correct,
            'next_correct': next_correct
        })

        page_contents=[]
        for page_index in range(prev_correct, next_correct+1):
            # Add bounds checking to prevent IndexError
            list_index = page_index - start_index
            if 0 <= list_index < len(page_list):
                page_text = f"<physical_index_{page_index}>\n{page_list[list_index][0]}\n<physical_index_{page_index}>\n\n"
                page_contents.append(page_text)
            else:
                continue
        content_range = ''.join(page_contents)

        physical_index_int = await single_toc_item_index_fixer(incorrect_item['title'], content_range)

        # Check if the result is correct
        check_item = incorrect_item.copy()
        check_item['physical_index'] = physical_index_int
        check_result = await check_title_appearance(check_item, page_list, start_index)

        return {
            'list_index': list_index,
            'title': incorrect_item['title'],
            'physical_index': physical_index_int,
            'is_valid': check_result['answer'] == 'yes'
        }

    # Process incorrect items concurrently
    tasks = [
        process_and_check_item(item)
        for item in incorrect_results
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for item, result in zip(incorrect_results, results):
        if isinstance(result, Exception):
            if logger:
                logger.error(f"Processing item {item} generated an exception: {result}")
            else:
                print(f"Processing item {item} generated an exception: {result}")
            continue
    results = [result for result in results if not isinstance(result, Exception)]

    # Update the toc_with_page_number with the fixed indices and check for any invalid results
    invalid_results = []
    for result in results:
        if result['is_valid']:
            # Add bounds checking to prevent IndexError
            list_idx = result['list_index']
            if 0 <= list_idx < len(toc_with_page_number):
                toc_with_page_number[list_idx]['physical_index'] = result['physical_index']
            else:
                # Index is out of bounds, treat as invalid
                invalid_results.append({
                    'list_index': result['list_index'],
                    'title': result['title'],
                    'physical_index': result['physical_index'],
                })
        else:
            invalid_results.append({
                'list_index': result['list_index'],
                'title': result['title'],
                'physical_index': result['physical_index'],
            })

    logger.info(f'incorrect_results_and_range_logs: {incorrect_results_and_range_logs}')
    logger.info(f'invalid_results: {invalid_results}')

    return toc_with_page_number, invalid_results



async def fix_incorrect_toc_with_retries(toc_with_page_number, page_list, incorrect_results, start_index=1, max_attempts=3, logger=None):
    if logger:
        logger.info('start fix_incorrect_toc_with_retries')
    else:
        print('start fix_incorrect_toc_with_retries')
    fix_attempt = 0
    current_toc = toc_with_page_number
    current_incorrect = incorrect_results

    while current_incorrect:
        if logger:
            logger.info(f"Fixing {len(current_incorrect)} incorrect results")
        else:
            print(f"Fixing {len(current_incorrect)} incorrect results")

        current_toc, current_incorrect = await fix_incorrect_toc(current_toc, page_list, current_incorrect, start_index, logger)

        fix_attempt += 1
        if fix_attempt >= max_attempts:
            logger.info("Maximum fix attempts reached")
            break

    return current_toc, current_incorrect




################### verify toc #########################################################
async def verify_toc(page_list, list_result, start_index=1, N=None, logger=None):
    if logger:
        logger.info('start verify_toc')
    else:
        print('start verify_toc')
    # Find the last non-None physical_index
    last_physical_index = None
    for item in reversed(list_result):
        if item.get('physical_index') is not None:
            last_physical_index = item['physical_index']
            break

    # Early return if we don't have valid physical indices
    if last_physical_index is None or last_physical_index < len(page_list)/2:
        return 0, []

    # Determine which items to check
    if N is None:
        if logger:
            logger.info('check all items')
        else:
            print('check all items')
        sample_indices = range(0, len(list_result))
    else:
        N = min(N, len(list_result))
        if logger:
            logger.info(f'check {N} items')
        else:
            print(f'check {N} items')
        sample_indices = random.sample(range(0, len(list_result)), N)

    # Prepare items with their list indices
    indexed_sample_list = []
    for idx in sample_indices:
        item = list_result[idx]
        # Skip items with None physical_index (these were invalidated by validate_and_truncate_physical_indices)
        if item.get('physical_index') is not None:
            item_with_index = item.copy()
            item_with_index['list_index'] = idx  # Add the original index in list_resul
            indexed_sample_list.append(item_with_index)

    # Run checks concurrently
    tasks = [
        check_title_appearance(item, page_list, start_index)
        for item in indexed_sample_list
    ]
    results = await asyncio.gather(*tasks)

    # Process results
    correct_count = 0
    incorrect_results = []
    for result in results:
        if result['answer'] == 'yes':
            correct_count += 1
        else:
            incorrect_results.append(result)

    # Calculate accuracy
    checked_count = len(results)
    accuracy = correct_count / checked_count if checked_count > 0 else 0
    if logger:
        logger.info(f"accuracy: {accuracy*100:.2f}%")
    else:
        print(f"accuracy: {accuracy*100:.2f}%")
    return accuracy, incorrect_results





################### main process #########################################################
async def meta_processor(page_list, mode=None, toc_content=None, toc_page_list=None, start_index=1, opt=None, logger=None, model=None):
    if logger:
        logger.info(f"meta_processor mode: {mode}")
    else:
        print(mode)
    if logger:
        logger.info(f"meta_processor start_index: {start_index}")
    else:
        print(f'start_index: {start_index}')

    if mode == 'process_toc_with_page_numbers':
        toc_with_page_number = await process_toc_with_page_numbers(toc_content, toc_page_list, page_list, toc_check_page_num=opt.toc_check_page_num, logger=logger) # Now async, model param removed
    elif mode == 'process_toc_no_page_numbers':
        toc_with_page_number = await process_toc_no_page_numbers(toc_content, toc_page_list, page_list, logger=logger) # Now async, model param removed
    else:
        # Pass model parameter to process_no_toc for agent model override
        toc_with_page_number = await process_no_toc(page_list, start_index=start_index, logger=logger, model=model)

    toc_with_page_number = [item for item in toc_with_page_number if item.get('physical_index') is not None]

    toc_with_page_number = validate_and_truncate_physical_indices(
        toc_with_page_number,
        len(page_list),
        start_index=start_index,
        logger=logger
    )

    accuracy, incorrect_results = await verify_toc(page_list, toc_with_page_number, start_index=start_index, logger=logger)

    logger.info({
        'mode': 'process_toc_with_page_numbers',
        'accuracy': accuracy,
        'incorrect_results': incorrect_results
    })
    if accuracy == 1.0 and len(incorrect_results) == 0:
        return toc_with_page_number
    if accuracy > 0.6 and len(incorrect_results) > 0:
        toc_with_page_number, incorrect_results = await fix_incorrect_toc_with_retries(toc_with_page_number, page_list, incorrect_results,start_index=start_index, max_attempts=3, logger=logger)
        return toc_with_page_number
    
    if mode == 'process_toc_with_page_numbers':
        return await meta_processor(page_list, mode='process_toc_no_page_numbers', toc_content=toc_content, toc_page_list=toc_page_list, start_index=start_index, opt=opt, logger=logger)
    
    if mode == 'process_toc_no_page_numbers':
        return await meta_processor(page_list, mode='process_no_toc', start_index=start_index, opt=opt, logger=logger)
    
    raise Exception('Processing failed')


async def process_large_node_recursively(node, page_list, opt=None, logger=None):
    node_page_list = page_list[node['start_index']-1:node['end_index']]
    token_num = sum([page[1] for page in node_page_list])

    if node['end_index'] - node['start_index'] > opt.max_page_num_each_node and token_num >= opt.max_token_num_each_node:
        if logger:
            logger.info(f"large node: {node['title']}, start_index: {node['start_index']}, end_index: {node['end_index']}, token_num: {token_num}")
        else:
            print('large node:', node['title'], 'start_index:', node['start_index'], 'end_index:', node['end_index'], 'token_num:', token_num)

        node_toc_tree = await meta_processor(node_page_list, mode='process_no_toc', start_index=node['start_index'], opt=opt, logger=logger)
        node_toc_tree = await check_title_appearance_in_start_concurrent(node_toc_tree, page_list, logger=logger)

        # Filter out items with None physical_index before post_processing
        valid_node_toc_items = [item for item in node_toc_tree if item.get('physical_index') is not None]

        if valid_node_toc_items and node['title'].strip() == valid_node_toc_items[0]['title'].strip():
            node['nodes'] = post_processing(valid_node_toc_items[1:], node['end_index'])
            node['end_index'] = valid_node_toc_items[1]['start_index'] if len(valid_node_toc_items) > 1 else node['end_index']
        else:
            node['nodes'] = post_processing(valid_node_toc_items, node['end_index'])
            node['end_index'] = valid_node_toc_items[0]['start_index'] if valid_node_toc_items else node['end_index']

    if 'nodes' in node and node['nodes']:
        tasks = [
            process_large_node_recursively(child_node, page_list, opt, logger=logger)
            for child_node in node['nodes']
        ]
        await asyncio.gather(*tasks)

    return node

async def tree_parser(page_list, opt, doc=None, logger=None, model=None):
    check_toc_result = await check_toc(page_list, opt, logger=logger)
    logger.info(check_toc_result)

    if check_toc_result.get("toc_content") and check_toc_result["toc_content"].strip() and check_toc_result["page_index_given_in_toc"] == "yes":
        toc_with_page_number = await meta_processor(
            page_list,
            mode='process_toc_with_page_numbers',
            start_index=1,
            toc_content=check_toc_result['toc_content'],
            toc_page_list=check_toc_result['toc_page_list'],
            opt=opt,
            logger=logger,
            model=model)
    else:
        toc_with_page_number = await meta_processor(
            page_list,
            mode='process_no_toc',
            start_index=1,
            opt=opt,
            logger=logger,
            model=model)

    toc_with_page_number = add_preface_if_needed(toc_with_page_number)
    toc_with_page_number = await check_title_appearance_in_start_concurrent(toc_with_page_number, page_list, logger=logger)

    # Filter out items with None physical_index before post-processing
    valid_toc_items = [item for item in toc_with_page_number if item.get('physical_index') is not None]

    toc_tree = post_processing(valid_toc_items, len(page_list))
    tasks = [
        process_large_node_recursively(node, page_list, opt, logger=logger)
        for node in toc_tree
    ]
    await asyncio.gather(*tasks)

    return toc_tree


async def page_index_main(doc, opt=None):
    logger = JsonLogger(doc)

    # If doc is a string, normalize the path to handle potential path issues
    if isinstance(doc, str):
        # Normalize path to resolve any redundant separators or relative path components
        normalized_path = os.path.normpath(doc)

        # Check if the normalized path exists and is a PDF
        is_valid_pdf = os.path.isfile(normalized_path) and normalized_path.lower().endswith(".pdf")

        # If valid after normalization, update doc to use the normalized path
        if is_valid_pdf:
            doc = normalized_path
    else:
        # For non-string inputs, check if it's a BytesIO objec
        is_valid_pdf = isinstance(doc, BytesIO)

    if not is_valid_pdf:
        # Provide more helpful error message
        if isinstance(doc, str):
            raise ValueError(f"PDF file not found or invalid: '{doc}'. Please check the path and ensure it's a valid PDF file.")
        
        raise ValueError("Unsupported input type. Expected a PDF file path or BytesIO object.")

    if logger:
        logger.info('Parsing PDF...')
    else:
        print('Parsing PDF...')
    page_list = get_page_tokens(doc)

    logger.info({'total_page_number': len(page_list)})
    logger.info({'total_token': sum([page[1] for page in page_list])})

    # Pass model parameter from opt to tree_parser for agent model override
    structure = await tree_parser(page_list, opt, doc=doc, logger=logger, model=opt.model if hasattr(opt, 'model') else None)
    if opt.if_add_node_id == 'yes':
        write_node_id(structure)
    if opt.if_add_node_text == 'yes':
        add_node_text(structure, page_list)
    if opt.if_add_node_summary == 'yes':
        if opt.if_add_node_text == 'no':
            add_node_text(structure, page_list)
        await generate_summaries_for_structure(structure, model=opt.model)
        if opt.if_add_node_text == 'no':
            remove_structure_text(structure)
        if opt.if_add_doc_description == 'yes':
            doc_description = await generate_doc_description(structure, model=opt.model)
            return {
                'doc_name': get_pdf_name(doc),
                'doc_description': doc_description,
                'structure': structure,
            }
    return {
        'doc_name': get_pdf_name(doc),
        'structure': structure,
    }


async def page_index(doc, model=None, toc_check_page_num=None, max_page_num_each_node=None,
               max_token_num_each_node=None, if_add_node_id=None, if_add_node_summary=None,
               if_add_doc_description=None, if_add_node_text=None):

    user_opt = {
        arg: value for arg, value in locals().items()
        if arg != "doc" and value is not None
    }
    opt = ConfigLoader().load(user_opt)
    return await page_index_main(doc, opt)


def validate_and_truncate_physical_indices(toc_with_page_number, page_list_length, start_index=1, logger=None):
    """
    Validates and truncates physical indices that exceed the actual document length.
    This prevents errors when TOC references pages that don't exist in the document (e.g. the file is broken or incomplete).
    """
    if not toc_with_page_number:
        return toc_with_page_number

    max_allowed_page = page_list_length + start_index - 1
    truncated_items = []

    for i, item in enumerate(toc_with_page_number):
        if item.get('physical_index') is not None:
            original_index = item['physical_index']
            if original_index > max_allowed_page:
                item['physical_index'] = None
                truncated_items.append({
                    'title': item.get('title', 'Unknown'),
                    'original_index': original_index
                })
                if logger:
                    logger.info(f"Removed physical_index for '{item.get('title', 'Unknown')}' (was {original_index}, too far beyond document)")

    if truncated_items and logger:
        logger.info(f"Total removed items: {len(truncated_items)}")

    print(f"Document validation: {page_list_length} pages, max allowed index: {max_allowed_page}")
    if truncated_items:
        print(f"Truncated {len(truncated_items)} TOC items that exceeded document length")

    return toc_with_page_number