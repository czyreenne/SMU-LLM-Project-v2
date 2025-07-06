# ----- REQUIRED IMPORTS -----

import os
import sys
import json
import datetime
import argparse
import subprocess
from typing import Dict, Optional, List
import multiprocessing
from helper.agent_clients import AgentClient
from helper.legalagents import LegalReviewPanel
from dotenv import load_dotenv
from helper.configloader import load_agent_config
from helper.markdown_translator import convert_to_md
# ----- INITIALIZATION CODE -----

load_dotenv()

# ----- HELPER FUNCTIONS -----

def process_hypothetical_directory(hypo_dir: str, selected_indices: Optional[List[int]] = None) -> str:
    """
    wrapper function that processes a directory of hypothetical pdfs and return the selected scenarios and questions
    """
    processed_dir = os.path.join("output")
    os.makedirs(processed_dir, exist_ok=True)
    print(f"\nExtracting hypotheticals from {hypo_dir}...")
    try:
        subprocess.run([sys.executable, "helper/extract_hypo.py", "--inpath", hypo_dir, "--outpath", processed_dir], check=True) # modified this to use current environment
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error running extract_hypo.py: {str(e)}")
    json_path = os.path.join(processed_dir, "extracted_data.json")
    if not os.path.exists(json_path):
        raise Exception(f"Expected output file {json_path} not found")
    with open(json_path, 'r') as f:
        extracted_data = json.load(f)
    if not extracted_data:
        raise Exception("No hypotheticals were extracted from the provided directory")
    
    # If indices are not provided (i.e., we're in a subprocess), use all hypotheticals
    if selected_indices is None:
        # Use all available hypotheticals
        selected_indices = list(range(1, len(extracted_data) + 1))
        print(f"\nUsing all available hypotheticals: {len(selected_indices)} files")
    else:
        print("\nAvailable hypotheticals:")
        for i, item in enumerate(extracted_data, 1):
            print(f"{i}. {item['file']} ({len(item['scenario'])} chars, {item['metadata']['num_pages']} pages)")
        
        # Interactive selection only if running in main process
        if not selected_indices:
            while not selected_indices:
                try:
                    selection = input("\nEnter the numbers of hypotheticals to analyze (comma-separated, e.g., '1,3,4') or press Enter for all: ")
                    if not selection.strip():  # If user just presses Enter
                        selected_indices = list(range(1, len(extracted_data) + 1))
                        print(f"Using all available hypotheticals: {len(selected_indices)} files")
                    else:
                        selected_indices = [int(idx.strip()) for idx in selection.split(",")]
                        if any(idx < 1 or idx > len(extracted_data) for idx in selected_indices): # indices validation but might not be necessary
                            print("Invalid selection. Please enter valid numbers.")
                            selected_indices = []
                except (ValueError, EOFError):
                    # Handle both invalid input and EOF (subprocess case)
                    print("Using all available hypotheticals due to input error.")
                    selected_indices = list(range(1, len(extracted_data) + 1))
                    break
    
    combined_scenario = ""
    combined_questions = []
    for idx in selected_indices:
        item = extracted_data[idx-1]
        combined_scenario += f"\n\n--- HYPOTHETICAL {idx}: {item['file']} ---\n\n{item['scenario']}"
        if item['questions']:
            combined_questions.extend([f"From {item['file']}: {q}" for q in item['questions']])
    analysis_text = combined_scenario # combine scenario and questions into single text
    if combined_questions:
        analysis_text += "\n\nQUESTIONS:\n" + "\n".join([f"{i+1}. {q}" for i, q in enumerate(combined_questions)])
    return analysis_text

class LegalSimulationWorkflow:
    def __init__(self, legal_question: str, api_keys: dict, model_backbone: Optional[str] = None, hypothetical: Optional[str] = None, hypothetical_indices: Optional[List[int]] = None, shared_results_dir: Optional[str] = None):
        """
        initialize the legal simulation workflow
        """
        self.legal_question = legal_question
        self.hypothetical = hypothetical
        self.hypothetical_indices = hypothetical_indices
        self.api_keys = api_keys
        self.model_backbone = model_backbone

        self.agent_configs = load_agent_config()

        # Initialize agents using AgentClient
        self.agents = {}
        for agent_name, config in self.agent_configs.items():
            self.agents[agent_name] = AgentClient(
                name=agent_name,
                config=self.agent_configs,
                agent_type=config["type"],
                model_str=self.model_backbone,
                api_keys=self.api_keys,
                allowed_collections=config["allowed_collections"] 
            )

        # Use shared results directory or create a new one
        if shared_results_dir:
            self.results_dir = shared_results_dir
            self.timestamp = os.path.basename(shared_results_dir).replace("analysis_", "")
        else:
            self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = os.path.join("results", f"analysis_{self.timestamp}")
        
        os.makedirs(self.results_dir, exist_ok=True)

    def _save_analysis_results(self, results: Dict) -> None:
        """
        save analysis results to a json file
        """
        # Use model name in filename to avoid conflicts (there may be better ways to handle this, 
        # can also put all models in a single file and use model name as the key, but then frontend will need to change too...)
        safe_model_name = self.model_backbone.replace("/", "_").replace("\\", "_").replace(":", "_")
        output_file = os.path.join(self.results_dir, f"analysis_results_{safe_model_name}.json")
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"[{self.model_backbone}] JSON results saved to: {output_file}")
            
            # Convert to markdown
            try:
                md_file = os.path.join(self.results_dir, f"analysis_results_{safe_model_name}.md")
                convert_to_md(output_file, md_file)
            except Exception as md_error:
                print(f"[{self.model_backbone}] Warning: Failed to convert to markdown: {str(md_error)}")
                # Don't fail the entire process if markdown conversion fails
                
        except Exception as e:
            raise Exception(f"Error saving analysis results: {str(e)}")

    def perform_legal_analysis(self) -> None:
        """
        execute the complete legal analysis workflow
        """

        try:
            print("\nInitiating legal analysis workflow...")


            if self.hypothetical:
                analysis_text = process_hypothetical_directory(self.hypothetical, self.hypothetical_indices)
                analysis_results = {
                    "legal_question": None,
                    "hypothetical": analysis_text,
                    "timestamp": self.timestamp,
                    "model": self.model_backbone,
                    "agent_outputs": {},
                    "final_synthesis": None
                }
                print(analysis_text)
            else:
                analysis_text = self.legal_question

                analysis_results = {
                    "legal_question": analysis_text,
                    "hypothetical": None,
                    "timestamp": self.timestamp,
                    "model": self.model_backbone,
                    "agent_outputs": {},
                    "final_synthesis": None
                }

            # Perform analysis for each agent
            for agent_name, agent in self.agents.items():
                print(f"\nPerforming analysis using {agent_name}...")
                agent_results = agent.perform_full_structured_analysis(question=analysis_text)
                analysis_results["agent_outputs"][agent_name] = agent_results

            # Synthesize reviews using Internal and External outputs
            print("\nSynthesizing perspectives...")
            internal_review = analysis_results["agent_outputs"]["internal"].get("review", "")
            external_review = analysis_results["agent_outputs"]["external"].get("review", "")

            reviews = [
                {"perspective": "internal_law", "review": internal_review},
                {"perspective": "external_law", "review": external_review}
            ]
            print('check1 ')
            review_panel = LegalReviewPanel(
                input_model=self.model_backbone,
                api_keys=self.api_keys,
                agent_config=self.agent_configs,
                max_steps=len(reviews),
            )
            synthesis = review_panel.synthesize_reviews(reviews, source_text=analysis_text)
            print('check end')
            analysis_results["final_synthesis"] = synthesis

            # Save all results
            print("\nSaving analysis results...")
            self._save_analysis_results(analysis_results)

            print(f"\nAnalysis complete! Results saved in: {self.results_dir}")

        except Exception as e:
            raise Exception(f"Error during legal analysis: {str(e)}")


def parse_arguments():
    """
    parse command-line arguments
    """
    parser = argparse.ArgumentParser(description="Legal Analysis Simulation System")
    parser.add_argument("--model", type=str, help="Selected model for generation")
    parser.add_argument("--question", type=str, help="The legal question to analyze")
    parser.add_argument("--hypo", type=str, help="Directory path containing hypothetical PDFs to analyze")
    return parser.parse_args()



def run_model_for_hypos(model: str, legal_question: str, hypothetical: str, api_keys: dict, hypothetical_indices: Optional[List[int]] = None, shared_results_dir: Optional[str] = None):
    """
    Run the actual LegalSimulationWorkflow for a given model.
    If an error occurs, it will continue with other models.
    """
    try:
        print(f"\n[{model}] Starting analysis...")
        workflow = LegalSimulationWorkflow(
            legal_question=legal_question,
            api_keys=api_keys,
            model_backbone=model,
            hypothetical=hypothetical,
            hypothetical_indices=hypothetical_indices,
            shared_results_dir=shared_results_dir
        )
        workflow.perform_legal_analysis()
        print(f"[{model}] ✓ Analysis completed successfully.")
    except Exception as e:
        print(f"[{model}] ✗ FAILED: {str(e)}")
        print(f"[{model}] Error details: {type(e).__name__}")
        # Log the full traceback for debugging
        import traceback
        print(f"[{model}] Full traceback:")
        traceback.print_exc()
        print(f"[{model}] Continuing with other models...")

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
    settings_path = os.path.join(os.path.dirname(__file__), 'settings/settings.json')
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
            print(f"\nExtracting hypotheticals from {hypothetical}...")
            subprocess.run([sys.executable, "helper/extract_hypo.py", "--inpath", hypothetical, "--outpath", processed_dir], check=True)
            json_path = os.path.join(processed_dir, "extracted_data.json")
            
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
            print(f"✓ Process for {models[i]} completed successfully")
        else:
            failed_count += 1
            print(f"✗ Process for {models[i]} failed with exit code {p.exitcode}")
    
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