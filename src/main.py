# ----- REQUIRED IMPORTS -----

import os
import sys
import json
import datetime
import argparse
import subprocess
from typing import Dict, Optional, List
import multiprocessing
from dotenv import load_dotenv
from helper.legalworkflow import (
    LegalSimulationWorkflow,
    run_individual_hypothetical_analysis,
    run_model_for_hypos
)

# ----- INITIALIZATION CODE -----

load_dotenv()

# ----- HELPER FUNCTIONS -----

def parse_arguments():
    """
    parse command-line arguments
    """
    parser = argparse.ArgumentParser(description="Legal Analysis Simulation System")
    parser.add_argument("--model", type=str, help="Selected model for generation")
    parser.add_argument("--question", type=str, help="The legal question to analyze")
    parser.add_argument("--hypo", type=str, help="Directory path containing hypothetical PDFs to analyze")
    return parser.parse_args()

def main():
    """
    main execution flow
    """
    args = parse_arguments()
    legal_question = args.question
    hypothetical = args.hypo
    selected_model = args.model

    if hypothetical and not os.path.isdir(hypothetical):
        raise ValueError(f"The specified hypothetical directory '{hypothetical}' does not exist or is not a directory.")
    if legal_question and hypothetical:
        raise ValueError("Cannot provide both a legal question and a hypothetical. Please choose one.")
    if not legal_question and not hypothetical:
        raise ValueError("Either a legal question (--question) or a legal hypothetical directory (--hypo) must be provided.")

    # Get API keys from environment variables
    api_keys = {
        'openai': os.getenv('OPENAI_API_KEY'),
        'deepseek': os.getenv('DEEPSEEK_API_KEY'),
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
    }

    if not any(api_keys.values()):
        raise ValueError("No API keys provided. At least one API key must be provided via environment variables.")
    else:
        missing_keys = [key for key, value in api_keys.items() if not value]
        if missing_keys:
            print(f"\nWarning: The following API keys are missing: {', '.join(missing_keys)}")
            print("Only services with valid API keys will be available.")
        available_keys = [key for key, value in api_keys.items() if value]
        print(f"\nAvailable API services: {', '.join(available_keys)}")

    # Load models from settings.json
    settings_path = os.path.join(os.path.dirname(__file__), 'settings', 'settings.json')
    with open(settings_path, 'r') as f:
        settings = json.load(f)
    all_models: List[str] = settings.get('models', [])
    if not all_models:
        raise ValueError("No models found in settings.json.")

    # If model is speficied via CLI
    if selected_model:
        # Run only the specified model
        if selected_model not in all_models:
            print(f"Warning: Specified model '{selected_model}' not found in settings.json")
            print(f"Available models: {', '.join(all_models)}")
            models = [selected_model]  # Still allow custom models
        else:
            models = [selected_model]
        print(f"\nRunning analysis for single model: {selected_model}")
    else:
        # Run all models if no specific model is provided
        models = all_models
        print(f"\nNo specific model provided. Running analysis for all models: {', '.join(models)}")

    # For interactive hypothetical selection, do it once in the main process
    hypothetical_indices = None
    if hypothetical:
        try:
            # Do interactive selection in main process first
            processed_dir = os.path.join("output")
            os.makedirs(processed_dir, exist_ok=True)

            json_path = os.path.join(processed_dir, "extracted_data.json")
            if not os.path.exists(json_path):
                print(f"\nNo extracted hypotheticals detected. Now extracting hypotheticals from {hypothetical}...")
                subprocess.run([sys.executable, "helper/extract_hypo.py", "--inpath", hypothetical, "--outpath", processed_dir], check=True)
            
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    extracted_data = json.load(f)
                
                if extracted_data:
                    print("\nAvailable hypotheticals:")
                    for i, item in enumerate(extracted_data, 1):
                        print(f"{i}. {item['file']} ({len(item['scenario'])} chars, {item['metadata']['num_pages']} pages)")
                    
                    while hypothetical_indices is None:
                        try:
                            selection = input("\nEnter the numbers of hypotheticals to analyze (comma-separated, e.g., '1,3,4') or press Enter for all: ")
                            if not selection.strip():  # If user just presses Enter
                                hypothetical_indices = list(range(1, len(extracted_data) + 1))
                                print(f"Using all available hypotheticals: {len(hypothetical_indices)} files")
                            else:
                                hypothetical_indices = [int(idx.strip()) for idx in selection.split(",")]
                                if any(idx < 1 or idx > len(extracted_data) for idx in hypothetical_indices):
                                    print("Invalid selection. Please enter valid numbers.")
                                    hypothetical_indices = None
                        except ValueError:
                            print("Invalid input. Please enter numbers separated by commas.")
                        except EOFError:
                            # Use all hypotheticals if input fails
                            hypothetical_indices = list(range(1, len(extracted_data) + 1))
                            print(f"Using all available hypotheticals: {len(hypothetical_indices)} files")
        except Exception as e:
            print(f"Error during hypothetical preprocessing: {e}")
            # Continue without pre-processing, let each subprocess handle it

    # Create shared results directory for all models
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    shared_results_dir = os.path.join("results", f"analysis_{timestamp}")
    os.makedirs(shared_results_dir, exist_ok=True)
    
    print(f"All results will be saved to: {shared_results_dir}")

    if hypothetical:
        # Individual hypothetical analysis mode
        print("\n=== INDIVIDUAL HYPOTHETICAL ANALYSIS MODE ===")
        
        # Run analysis for each selected hypothetical individually
        for hypo_index in hypothetical_indices:
            print(f"Processing hypothetical {hypo_index}...")
            run_individual_hypothetical_analysis(
                hypo_index=hypo_index,
                hypo_dir=hypothetical,
                legal_question=legal_question or "",
                api_keys=api_keys,
                shared_results_dir=shared_results_dir,
                models=models
            )
        
        print(f"\n=== ANALYSIS COMPLETE ===")
        print(f"Individual hypothetical results saved to: {shared_results_dir}")
        
    else:
        # Original combined analysis mode for legal questions
        print("\n=== COMBINED ANALYSIS MODE (Legal Question) ===")
        
        # Use multiprocessing to run all models in parallel
        processes = []
        for model in models:
            p = multiprocessing.Process(target=run_model_for_hypos, args=(model, legal_question or "", hypothetical or "", api_keys, hypothetical_indices, shared_results_dir))
            p.start()
            processes.append(p)
        
        # Wait for all processes to complete
        completed_count = 0
        failed_count = 0
        for i, p in enumerate(processes):
            p.join()
            if p.exitcode == 0:
                completed_count += 1
                print(f"✓ {models[i]} completed")
            else:
                failed_count += 1
                print(f"✗ {models[i]} failed")
        
        print(f"\n=== SUMMARY ===")
        print(f"Total models: {len(models)}")
        print(f"Completed successfully: {completed_count}")
        print(f"Failed: {failed_count}")
        if failed_count > 0:
            print(f"Note: Failed models may be due to missing API keys or other configuration issues.")
        print(f"All results saved to: {shared_results_dir}")
        print(f"===============")

# ----- EXECUTION CODE -----

if __name__ == "__main__":
    main()
