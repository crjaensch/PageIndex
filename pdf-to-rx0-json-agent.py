#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pathlib",
#     "click",
#     "openai>=1.12.0",
#     "openai-agents",
# ]
# ///

"""
Extract data from PDF documents according to a JSON schema derived from a digital product description
using the OpenAI Agents SDK approach.
"""

import json
import base64
import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Any
import click

try:
    from agents import Agent, Runner, set_default_openai_api
except ImportError:
    print("Required packages not found. Please run this script with uv:")
    print("uv run pdf-to-rx0-json-agent.py")
    sys.exit(1)


class ProcessContext:
    """Context for PDF processing."""
    def __init__(self, pdf_content: bytes, schema_data: List[Dict[str, Any]], pdf_path: Path, silent: bool = False, model: str = "gpt-4o"):
        self.pdf_content = pdf_content
        self.pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
        self.schema_data = schema_data
        self.schema_str = json.dumps(schema_data, indent=2)
        self.pdf_path = pdf_path
        self.model = model
        self.silent = silent
        self.extraction_result = None
        self.validation_result = None
        self.grounding_result = None


def process_node(node):
    """Process a single node, keeping only the specified properties."""
    if node.get("type") == "number":
        # Extract only the properties we want
        result = {
            "type": "number",
            "name": node.get("name"),
            "labelInt": node.get("labelInt")
        }
        
        # Add optional properties if they exist
        if "kind" in node:
            result["kind"] = node["kind"]
        if "requiredExp" in node:
            result["requiredExp"] = node["requiredExp"]
            
        return result
    elif node.get("type") == "section":
        # For sections, return None to skip them in the main extract_nodes function
        # The extract_nodes function will handle processing the section's subnodes
        return None
    elif node.get("type") == "group":
        # Process groups
        group_result = {
            "type": "group",
            "name": node.get("name"),
            "labelInt": node.get("labelInt"),
            "nodes": []
        }
        
        # Process each node in the group
        for child in node.get("nodes", []):
            if child.get("type") == "number":
                number_node = process_node(child)
                if number_node:
                    group_result["nodes"].append(number_node)
        
        return group_result
    
    return None


def extract_nodes(json_data):
    """Extract nodes of type 'number' or 'section' from the calculation node."""
    result = []
    
    # Get the calculation nodes
    calculation_nodes = json_data.get("calculation", {}).get("nodes", [])
    
    # Process each node
    for node in calculation_nodes:
        if node.get("type") == "section":
            # For sections, process their subnodes directly
            for child in node.get("nodes", []):
                processed_child = process_node(child)
                if processed_child:
                    result.append(processed_child)
        else:
            # For non-section nodes, process them normally
            processed_node = process_node(node)
            if processed_node:
                result.append(processed_node)
    
    return result


def extract_json_from_text(text: str) -> str:
    """Extract JSON string from text that might contain markdown or other formatting."""
    # Check if the text contains a code block
    if "```json" in text:
        # Extract content between ```json and ```
        start = text.find("```json") + 7
        end = text.find("```", start)
        return text[start:end].strip()
    elif "```" in text:
        # Extract content between ``` and ```
        start = text.find("```") + 3
        end = text.find("```", start)
        return text[start:end].strip()
    
    # If no code block, try to find JSON object directly
    # Look for the first { and the last }
    start = text.find("{")
    end = text.rfind("}") + 1
    
    if start >= 0 and end > start:
        return text[start:end].strip()
    
    # If all else fails, return the original text
    return text


async def extract_data_from_pdf(context: ProcessContext) -> Dict[str, Any]:
    """Extract data from PDF using OpenAI Agent."""
    pdf_name = context.pdf_path.name

    # Create an agent for PDF extraction
    agent = Agent(
        name="PDF Data Extraction Agent",
        instructions="You are an expert in extracting specific financial KPIs from PDF documents. Your task is to extract data according to the provided schema and transform it into a flattened JSON structure. For any data that cannot be found with high confidence, mark it as null and continue with the extraction.",
        model=context.model,
    )
    
    # Create the input with the PDF file
    input_with_pdf = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"Extract financial data from the PDF document '{pdf_name}'.\n\nThe schema below describes the hierarchical structure of the data:\n{context.schema_str}\n\nTransform this into a flattened JSON object where:\n1. Each 'group' becomes a top-level key in the output\n2. Each 'number' within a group becomes a property of that group\n3. Use the 'name' field as the property name\n4. The value should be the actual numeric value from the PDF\n\nExample format (with made-up data):\n{{\n  \"GroupName1\": {{\n    \"metric1\": 123.45,\n    \"metric2\": 67.89\n  }},\n  \"GroupName2\": {{\n    \"metric3\": 100.0,\n    \"metric4\": null  // if data couldn't be found\n  }}\n}}\n\nReturn ONLY a valid JSON object with the actual values from the PDF. Do not include any additional text or explanation, just the JSON object."
                },
                {
                    "type": "input_file",
                    "filename": pdf_name,
                    "file_data": f"data:application/pdf;base64,{context.pdf_base64}"
                }
            ]
        }
    ]
    
    # Run the agent with the PDF input
    result = await Runner.run(agent, input=input_with_pdf)
    
    # Extract the JSON response
    response_text = result.final_output
    
    # Parse the JSON from the response text
    json_str = extract_json_from_text(response_text)
    
    try:
        extracted_data = json.loads(json_str)
        return extracted_data
    except json.JSONDecodeError:
        if not context.silent:
            print("Failed to parse JSON response. Raw response:")
            print(response_text)
        return {"error": "Failed to parse response"}


async def validate_extracted_data(context: ProcessContext) -> Dict[str, Any]:
    """Validate the extracted data against the schema."""

    # Create an agent for data validation
    agent = Agent(
        name="Data Validation Agent",
        instructions="You are an expert in validating data structures against schemas. Your task is to verify that the extracted JSON data conforms to the expected schema structure. Perform a thorough validation checking:\n1. All required fields are present\n2. Data types match expected types\n3. Structure follows the hierarchical organization defined in the schema.",
        model=context.model,
    )

    # Create the input for validation
    input_for_validation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"Validate the following extracted data against the schema:\n\nExtracted data:\n{json.dumps(context.extraction_result, indent=2)}\n\nSchema:\n{context.schema_str}\n\nReturn a JSON object with the following structure:\n{{\n  \"is_valid\": boolean,  // true if no critical errors found\n  \"validation_summary\": \"Brief summary of validation results\",\n  \"issues\": [\n    {{\n      \"path\": \"path.to.field\",  // Use dot notation to identify the field\n      \"expected\": \"what was expected\",\n      \"found\": \"what was found\",\n      \"severity\": \"error|warning\",  // Use 'error' for critical issues, 'warning' for minor issues\n      \"message\": \"Detailed explanation of the issue\"\n    }}\n  ]\n}}\n\nCritical errors include:\n- Missing required fields\n- Incorrect data types\n- Structural inconsistencies\n\nWarnings include:\n- Unexpected formatting\n- Potential inconsistencies between related fields\n\nIf no issues are found, return an empty array for 'issues'."
                }
            ]
        }
    ]
    
    # Run the agent with the validation input
    result = await Runner.run(agent, input=input_for_validation)
    
    # Extract the JSON response
    response_text = result.final_output
    json_str = extract_json_from_text(response_text)
    
    try:
        validation_result = json.loads(json_str)
        return validation_result
    except json.JSONDecodeError:
        if not context.silent:
            print("Failed to parse validation JSON. Raw response:")
            print(response_text)
        return {"error": "Failed to parse validation response"}


async def check_data_grounding(context: ProcessContext) -> Dict[str, Any]:
    """Check if the extracted data is grounded in the PDF content."""
    pdf_name = context.pdf_path.name

    # Create an agent for data grounding
    agent = Agent(
        name="Data Grounding Agent",
        instructions="You are an expert in verifying that extracted data is actually present in source documents. Your task is to check that the values in the extracted JSON data are actually supported by the content of the PDF. Perform a thorough grounding assessment by:\n1. Verifying numerical values match what's in the PDF\n2. Confirming categories and labels correspond to actual sections in the PDF\n3. Checking that no data has been hallucinated or fabricated\n4. Providing specific page numbers or text excerpts as evidence.",
        model=context.model,
    )
    
    # Create the input with the PDF file and extracted data
    input_with_pdf = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"Check if the following extracted data is grounded in the PDF content:\n\nExtracted data:\n{json.dumps(context.extraction_result, indent=2)}\n\nReturn a JSON object with the following structure:\n{{\n  \"is_fully_grounded\": boolean,  // true if all data is grounded in the PDF\n  \"grounding_summary\": \"Brief overall assessment of the extraction quality\",\n  \"ungrounded_items\": [\n    {{\n      \"path\": \"path.to.field\",  // Use dot notation to identify the field\n      \"extracted_value\": \"value that was extracted\",\n      \"confidence\": 0-100,  // Confidence percentage that this is incorrect\n      \"evidence\": \"Specific evidence from the PDF or explanation of why this is ungrounded\",\n      \"suggestion\": \"Optional suggested correction based on the PDF content\"\n    }}\n  ]\n}}\n\nA value is considered 'grounded' when:\n- Numerical values match exactly or are within reasonable rounding error\n- Text values match the meaning in the PDF, even if not verbatim\n- The structure accurately represents the organization in the PDF\n\nIf all data is properly grounded, return an empty array for 'ungrounded_items'."
                },
                {
                    "type": "input_file",
                    "filename": pdf_name,
                    "file_data": f"data:application/pdf;base64,{context.pdf_base64}"
                }
            ]
        }
    ]
    
    # Run the agent with the PDF input
    result = await Runner.run(agent, input=input_with_pdf)
    
    # Extract the JSON response
    response_text = result.final_output
    json_str = extract_json_from_text(response_text)
    
    try:
        grounding_result = json.loads(json_str)
        return grounding_result
    except json.JSONDecodeError:
        if not context.silent:
            print("Failed to parse grounding JSON. Raw response:")
            print(response_text)
        return {"error": "Failed to parse grounding response"}


@click.command()
@click.option(
    "--pdf",
    required=True,
    help="Path to the PDF file to process.",
    type=click.Path(exists=True),
)
@click.option(
    "--product",
    required=True,
    help="Path to the digital product description JSON file.",
    type=click.Path(exists=True),
)
@click.option(
    "--output",
    "-o",
    help="Output file path for the result. If not provided, prints to stdout.",
    type=click.Path(),
)
@click.option(
    "--schema-output",
    "-s",
    help="Output file path for the intermediate schema. If not provided, the schema is not saved.",
    type=click.Path(),
)
@click.option(
    "--api-key",
    help="OpenAI API key. If not provided, will use OPENAI_API_KEY environment variable.",
    envvar="OPENAI_API_KEY",
)
@click.option(
    "--model",
    help="OpenAI model to use.",
    default="gpt-4o",
)
@click.option(
    "--silent",
    is_flag=True,
    help="Suppress all output except for the requested data.",
)
@click.option(
    "--skip-validation",
    is_flag=True,
    help="Skip the validation and grounding checks.",
)
def pdf_to_json(
    pdf,
    product,
    output,
    schema_output,
    api_key,
    model,
    silent,
    skip_validation,
):
    """Process a PDF file using OpenAI's Agent SDK and extract data according to a JSON schema.

    This tool extracts specific data from a PDF file according to a JSON schema derived
    from a digital product description. The extracted data is returned as a JSON file
    conforming to the provided schema.
    """
    # Use asyncio.run to handle the async execution
    asyncio.run(_pdf_to_json_async(
        pdf, product, output, schema_output, api_key, model, silent, skip_validation
    ))


async def _pdf_to_json_async(
    pdf,
    product,
    output,
    schema_output,
    api_key,
    model,
    silent,
    skip_validation,
):
    """Async implementation of the PDF processing logic using the Agent SDK."""
    # Validate options
    if not api_key:
        raise click.ClickException(
            "No API key provided and OPENAI_API_KEY environment variable not set."
        )

    # Set up the OpenAI API to be used by the Agent SDK
    set_default_openai_api("responses")

    # Read the PDF file
    pdf_path = Path(pdf)
    if not silent:
        click.echo(f"Reading PDF file: {pdf_path.name}...", err=True)
    
    try:
        pdf_content = pdf_path.read_bytes()
    except Exception as e:
        raise click.ClickException(f"Error reading PDF file: {e}")

    # Read the product JSON file
    product_path = Path(product)
    if not silent:
        click.echo(f"Reading product JSON file: {product_path.name}...", err=True)
    
    try:
        with open(product_path, 'r') as f:
            product_data = json.load(f)
    except Exception as e:
        raise click.ClickException(f"Error reading product JSON file: {e}")

    # Extract the data nodes to create a simplified schema
    if not silent:
        click.echo("Extracting data nodes to create simplified schema...", err=True)
    
    schema_data = extract_nodes(product_data)
    
    # Save the schema if requested
    if schema_output:
        schema_output_path = Path(schema_output)
        if not silent:
            click.echo(f"Writing schema to {schema_output_path}...", err=True)
        with open(schema_output_path, 'w') as f:
            json.dump(schema_data, f, indent=2)

    # Initialize the processing context
    context = ProcessContext(
        pdf_content=pdf_content,
        schema_data=schema_data,
        pdf_path=pdf_path,
        model=model,
        silent=silent
    )

    try:
        # Extract data from PDF
        if not silent:
            click.echo(f"Extracting data from PDF using {model}...", err=True)
        
        extraction_result = await extract_data_from_pdf(context)
        context.extraction_result = extraction_result
        
        # If validation is not skipped, validate the extracted data
        if not skip_validation:
            # Validate the extracted data
            if not silent:
                click.echo("Validating extracted data...", err=True)
            
            validation_result = await validate_extracted_data(context)
            context.validation_result = validation_result
            
            # Check if the extracted data is grounded in the PDF content
            if not silent:
                click.echo("Checking if extracted data is grounded in PDF content...", err=True)
            
            grounding_result = await check_data_grounding(context)
            context.grounding_result = grounding_result
        
        # Prepare the final result
        if skip_validation or (context.validation_result and context.grounding_result):
            result_json = context.extraction_result
            
            # Add validation and grounding results if available
            if not skip_validation:
                result_json = {
                    "data": context.extraction_result,
                    "validation": context.validation_result,
                    "grounding": context.grounding_result
                }
        else:
            result_json = {
                "data": context.extraction_result,
                "validation": context.validation_result or {"error": "Validation failed"},
                "grounding": context.grounding_result or {"error": "Grounding check failed"}
            }

        # Output the result
        if output:
            output_path = Path(output)
            if not silent:
                click.echo(f"Writing output to {output_path}...", err=True)
            with open(output_path, 'w') as f:
                json.dump(result_json, f, indent=2)
        else:
            # Print to stdout
            print(json.dumps(result_json, indent=2))

    except Exception as e:
        raise click.ClickException(f"Error processing with OpenAI: {e}")


if __name__ == "__main__":
    pdf_to_json()
