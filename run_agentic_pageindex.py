#!/usr/bin/env python3
import asyncio
import os
import json
import click
from pathlib import Path
from pageindex.multi_agent_orchestrator import multi_agent_page_index
from pageindex.utils import ConfigLoader

@click.command()
@click.option('--pdf-path', required=True, type=click.Path(exists=True), help='Path to the PDF file')
@click.option('--model', default='gpt-4.1-mini', help='Model to use (e.g., gpt-4.1-mini, gpt-4o, gpt-3.5-turbo)')
@click.option('--toc-check-pages', default=20, type=int, help='Number of pages to check for table of contents')
@click.option('--max-pages-per-node', default=10, type=int, help='Maximum number of pages per node')
@click.option('--max-tokens-per-node', default=20000, type=int, help='Maximum number of tokens per node')
@click.option('--accuracy-threshold', default=0.8, type=float, help='Accuracy threshold for orchestrated processing')
@click.option('--include-summaries', is_flag=True, help='Include summaries in the output')
@click.option('--include-descriptions', is_flag=True, help='Include descriptions in the output')
@click.option('--include-text', is_flag=True, help='Include text content in the output')
@click.option('--timeout-minutes', default=10, type=int, help='Timeout for the orchestrated processing in minutes')
def main(pdf_path, model, toc_check_pages, max_pages_per_node, max_tokens_per_node, 
         accuracy_threshold, include_summaries, include_descriptions, 
         include_text, timeout_minutes):
    """
    Process PDF document using the multi-agent page index algorithm.
    
    This tool analyzes PDF documents and generates a structured representation
    of their content using a multi-agent orchestrated approach.
    """
    # Configure options
    config_loader = ConfigLoader()
    user_args_dict = {
        "model": model,
        "toc_check_page_num": toc_check_pages,
        "max_page_num_each_node": max_pages_per_node,
        "max_token_num_each_node": max_tokens_per_node,
        "include_node_id": "yes",  # Always generate node IDs
        "include_node_summary": "yes" if include_summaries else "no",
        "include_doc_description": "yes" if include_descriptions else "no",
        "include_node_text": "yes" if include_text else "no",
        # Add the additional parameters that were previously in requirements
        "accuracy_threshold": accuracy_threshold,
        "timeout_minutes": timeout_minutes
    }
    opt = config_loader.load(user_args_dict)
    
    # Process the PDF
    click.echo(f"Processing PDF: {pdf_path} with model: {model}")
    click.echo("Starting multi-agent page index analysis...")
    
    # Define the async processing function
    async def process():
        try:
            # Pass the parameters to the new multi_agent_page_index function
            # The opt object now contains all necessary configuration
            result = await multi_agent_page_index(pdf_path, opt=opt, model=model)
            click.echo('Parsing done, saving to file...')
            
            # Save results
            pdf_name = Path(pdf_path).stem
            os.makedirs('./results', exist_ok=True)
            
            output_file = f'./results/{pdf_name}_multi_agent_structure.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2)
            
            click.echo(f"Results saved to {output_file}")
            
        except Exception as e:
            click.echo(f"Error processing PDF: {str(e)}", err=True)
            raise
    
    # Run the async function
    asyncio.run(process())

if __name__ == "__main__":
    main()
