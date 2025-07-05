# ----- REQUIRED IMPORTS -----

import os
import sys
import json
import datetime
import argparse
import subprocess
from typing import Dict, Optional
from helper.agent_clients import AgentClient
from helper.legalagents import LegalReviewPanel
from dotenv import load_dotenv
from helper.configloader import load_agent_config
from helper.markdown_translator import convert_to_md
# ----- INITIALIZATION CODE -----

load_dotenv()

# ----- HELPER FUNCTIONS -----

def process_hypothetical_directory(hypo_dir: str) -> str:
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
    print("\nAvailable hypotheticals:")
    for i, item in enumerate(extracted_data, 1):
        print(f"{i}. {item['file']} ({len(item['scenario'])} chars, {item['metadata']['num_pages']} pages)")
    selected_indices = []
    while not selected_indices:
        selection = input("\nEnter the numbers of hypotheticals to analyze (comma-separated, e.g., '1,3,4'): ")
        try:
            selected_indices = [int(idx.strip()) for idx in selection.split(",")]
            if any(idx < 1 or idx > len(extracted_data) for idx in selected_indices): # indices validation but might not be necessary
                print("Invalid selection. Please enter valid numbers.")
                selected_indices = []
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")
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
    def __init__(self, legal_question: str, api_keys: dict, model_backbone: Optional[str] = None, hypothetical: Optional[str] = None):
        """
        initialize the legal simulation workflow
        """
        self.legal_question = legal_question
        self.hypothetical = hypothetical
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

        # Create results directory with timestamp
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join("results", f"analysis_{self.timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)

    def _save_analysis_results(self, results: Dict) -> None:
        """
        save analysis results to a json file
        """
        output_file = os.path.join(self.results_dir, "analysis_results.json")
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            convert_to_md(output_file)
        except Exception as e:
            raise Exception(f"Error saving analysis results: {str(e)}")

    def perform_legal_analysis(self) -> None:
        """
        execute the complete legal analysis workflow
        """

        try:
            print("\nInitiating legal analysis workflow...")


            if self.hypothetical:
                analysis_text = process_hypothetical_directory(self.hypothetical)
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

            # right now logic is just simplified to check for hypos first but
            # this is because im edge guarding within the runmac.sh and runwin.bat
            # calls already so that only one analysistext source can be called at 
            # once
            # ~ gong
            
            
            # changed this as its passing the hypo directory instead of the acutal hypo                 
            
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


def main():
    """
    main execution flow
    """

    args = parse_arguments()
    selected_model = args.model or "gpt-4o-mini"
    legal_question = args.question
    hypothetical = args.hypo

    # more logging, can remove if deemed unhelpful ~ gong
    if hypothetical and not os.path.isdir(hypothetical): # verify hypothetical directory exists if provided, actually im checking this in the runmac.sh and runwin.bat already so maybe can remove??? ~ gong
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
    else:  # just me being extra and adding more logging, we can remove this later ~ gong
        missing_keys = [key for key, value in api_keys.items() if not value]
        if missing_keys:
            print(f"\nWarning: The following API keys are missing: {', '.join(missing_keys)}")
            print("Only services with valid API keys will be available.")        
        available_keys = [key for key, value in api_keys.items() if value]
        print(f"\nAvailable API services: {', '.join(available_keys)}")

    try:
        print("\nInitializing legal simulation workflow...")
        workflow = LegalSimulationWorkflow(
            legal_question=legal_question or "", # pass empty string if none since guarding above alr ~ gong
            api_keys=api_keys,
            model_backbone=selected_model,
            hypothetical=hypothetical or "", # pass empty string if none since prev edge guarding should be good enough ~ gong
        )
        workflow.perform_legal_analysis()
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")

# ----- EXECUTION CODE -----

if __name__ == "__main__":
    main()