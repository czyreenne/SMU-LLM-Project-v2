import os
import json
import datetime
import sys
import subprocess
import multiprocessing
from typing import Dict, Optional, List
from .agent_clients import AgentClient
# from ..archive.legalagents import LegalReviewPanel
from .configloader import load_agent_config
from .markdown_translator import create_individual_analysis_files


def process_single_hypothetical(hypo_dir: str, hypo_index: int) -> tuple:
    """
    Process a single hypothetical by index and return its content and metadata
    """
    processed_dir = os.path.join("output")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Run extraction if not already done
    json_path = os.path.join(processed_dir, "extracted_data.json")
    if not os.path.exists(json_path):
        print(f"\nExtracting hypotheticals from {hypo_dir}...")
        try:
            subprocess.run([sys.executable, "src/helper/extract_hypo.py", "--inpath", hypo_dir, "--outpath", processed_dir], check=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"Error running extract_hypo.py: {str(e)}")
    
    if not os.path.exists(json_path):
        raise Exception(f"Expected output file {json_path} not found")
    
    with open(json_path, 'r') as f:
        extracted_data = json.load(f)
    
    if not extracted_data or hypo_index < 1 or hypo_index > len(extracted_data):
        raise Exception(f"Invalid hypothetical index {hypo_index}")
    
    item = extracted_data[hypo_index-1]
    
    # Create analysis text for this specific hypothetical
    scenario_text = f"\n\n--- HYPOTHETICAL {hypo_index}: {item['file']} ---\n\n{item['scenario']}"
    
    analysis_tasks = []
    if item['questions']:
        for i, q in enumerate(item['questions']):
            question_text = f"From {item['file']}: {q}"
            analysis_text = f"{scenario_text}\n\nQUESTION: {question_text}"
            analysis_tasks.append({
                "text": analysis_text,
                "question_index": i + 1
            })
    else:
        # If no questions, create one task with just the scenario
        analysis_tasks.append({
            "text": scenario_text,
            "question_index": 0 # 0 for scenario-only analysis
        })

    # Return analysis text and safe filename
    safe_filename = item['file'].replace('.pdf', '').replace(' ', '_').replace('-', '_')
    return analysis_tasks, safe_filename, item['file']


def save_individual_hypothetical_results(hypo_results: Dict, hypo_filename: str, shared_results_dir: str) -> None:
    """
    Save results for a single hypothetical across all models with enhanced structure
    """
    # Create JSON file with all models' results for this hypothetical
    hypo_json_file = os.path.join(shared_results_dir, f"{hypo_filename}.json")
    
    try:
        with open(hypo_json_file, 'w') as f:
            json.dump(hypo_results, f, indent=2)
        print(f"Hypothetical results saved to: {hypo_json_file}")
        
        # Create individual markdown files for each model and each question
        for i, question_result in enumerate(hypo_results.get("results_per_question", [])):
            is_first = (i == 0)
            question_index = question_result.get("question_index", "scenario")
            for model_results in question_result.get("models", []):
                if isinstance(model_results, dict) and "error" not in model_results:
                    model_name = model_results.get("model", "unknown_model")
                    try:
                        # Use enhanced markdown translator to create individual files
                        create_individual_analysis_files(
                            results=model_results,
                            base_output_dir=shared_results_dir,
                            model_name=model_name,
                            hypo_name=hypo_filename,
                            agent_config=load_agent_config(),
                            is_first_question=is_first
                        )
                        print(f"Enhanced markdown files created for {model_name} for question {question_index}")
                    except Exception as md_error:
                        print(f"Warning: Failed to create enhanced markdown for {model_name} for question {question_index}: {str(md_error)}")
                else:
                    model_name = model_results.get("model", "unknown_model")
                    print(f"Skipping markdown generation for {model_name} for question {question_index} due to errors in results")
                    
    except Exception as e:
        print(f"Error saving hypothetical results: {str(e)}")


def run_individual_hypothetical_analysis(hypo_index: int, hypo_dir: str, legal_question: str, api_keys: dict, shared_results_dir: str, models: List[str]):
    """
    Run analysis for a single hypothetical across all models
    """
    try:
        # Get hypothetical content
        analysis_tasks, hypo_filename, hypo_original_name = process_single_hypothetical(hypo_dir, hypo_index)
        
        print(f"\n=== ANALYZING HYPOTHETICAL {hypo_index}: {hypo_original_name} ===")
        
        # Dictionary to store results from all models for this hypothetical
        hypo_results = {
            "hypothetical_info": {
                "original_name": hypo_original_name,
                "index": hypo_index,
                "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            },
            "results_per_question": []
        }
        
        # Process each question/task for the hypothetical
        for task in analysis_tasks:
            analysis_text = task["text"]
            question_index = task["question_index"]
            
            print(f"\n--- Analyzing Question {question_index} ---")
            
            # Run all models for this task in parallel
            processes = []
            manager = multiprocessing.Manager()
            results_dict = manager.dict()
            
            for model in models:
                p = multiprocessing.Process(
                    target=run_single_model_for_hypothetical,
                    args=(model, analysis_text, api_keys, results_dict)
                )
                p.start()
                processes.append(p)
            
            # Wait for all models to complete
            for p in processes:
                p.join()
            
            # Collect results
            model_results_list = []
            for model in models:
                if model in results_dict:
                    model_results_list.append(dict(results_dict[model]))
                else:
                    model_results_list.append({"model": model, "error": "Analysis failed", "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")})

            hypo_results["results_per_question"].append({
                "question_index": question_index,
                "question_text": analysis_text.split("QUESTION:")[1].strip() if "QUESTION:" in analysis_text else "Scenario Analysis",
                "models": model_results_list
            })

        # Save results for this hypothetical
        save_individual_hypothetical_results(hypo_results, hypo_filename, shared_results_dir)
        
        print(f"✓ Completed analysis for hypothetical {hypo_index}: {hypo_original_name}")
        
    except Exception as e:
        print(f"✗ Failed to analyze hypothetical {hypo_index}: {str(e)}")


def run_single_model_for_hypothetical(model: str, analysis_text: str, api_keys: dict, results_dict):
    """
    Run a single model analysis for a hypothetical and store result
    """
    try:
        # Directly create agents instead of the full workflow to avoid creating directories
        agent_configs = load_agent_config()
        agents = {}
        for agent_name, config in agent_configs.items():
            agents[agent_name] = AgentClient(
                name=agent_name,
                config=agent_configs,
                agent_type=config["type"],
                model_str=model,
                api_keys=api_keys,
                allowed_collections=config["allowed_collections"] 
            )

        # Manually set the analysis text and run analysis
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_results = {
            "legal_question": None,
            "hypothetical": analysis_text,
            "timestamp": timestamp,
            "model": model,
            "agent_outputs": {},
            "final_synthesis": None
        }

        # Perform analysis for each agent
        for i, (agent_name, agent) in enumerate(agents.items()):
            agent_results = agent.perform_full_structured_analysis(question=analysis_text)
            print(f"[{model}] Analysis completed for agent: {agent_name} (Step {i+1}/{len(agents)})")
            analysis_results["agent_outputs"][agent_name] = agent_results
        
        # Store results in shared dictionary
        results_dict[model] = analysis_results
        
    except Exception as e:
        results_dict[model] = {"error": str(e), "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}


def run_model_for_hypos(model: str, legal_question: str, hypothetical: str, api_keys: dict, hypothetical_indices: Optional[List[int]] = None, shared_results_dir: Optional[str] = None):
    """
    Run the actual LegalSimulationWorkflow for a given model.
    If an error occurs, it will continue with other models.
    """
    try:
        workflow = LegalSimulationWorkflow(
            legal_question=legal_question,
            api_keys=api_keys,
            model_backbone=model,
            hypothetical=hypothetical,
            hypothetical_indices=hypothetical_indices,
            shared_results_dir=shared_results_dir
        )
        workflow.perform_legal_analysis()
        
    except Exception as e:
        print(f"[{model}] ✗ FAILED: {str(e)}")
        print(f"[{model}] Error details: {type(e).__name__}")
        # Log the full traceback for debugging
        import traceback
        print(f"[{model}] Full traceback:")
        traceback.print_exc()
        print(f"[{model}] Continuing with other models...")


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
        save analysis results to a json file and create enhanced markdown files
        """
        # Use model name in filename to avoid conflicts
        safe_model_name = self.model_backbone.replace("/", "_").replace("\\", "_").replace(":", "_")
        output_file = os.path.join(self.results_dir, f"analysis_results_{safe_model_name}.json")
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"[{self.model_backbone}] JSON results saved to: {output_file}")
            
            # Create enhanced markdown files using new structure
            try:
                create_individual_analysis_files(
                    results=results,
                    base_output_dir=self.results_dir,
                    model_name=self.model_backbone
                )
                print(f"[{self.model_backbone}] Enhanced markdown files created")
            except Exception as md_error:
                print(f"[{self.model_backbone}] Warning: Failed to create enhanced markdown: {str(md_error)}")
                # Don't fail the entire process if markdown conversion fails
                
        except Exception as e:
            raise Exception(f"Error saving analysis results: {str(e)}")

    def perform_legal_analysis(self) -> None:
        """
        execute the complete legal analysis workflow
        """
        print(self.agents)
        try:
            print("\nInitiating legal analysis workflow...")

            if self.hypothetical:
                # This path is now handled by run_individual_hypothetical_analysis
                print("Hypothetical analysis is handled by the main script.")
                return

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

            # Save all results
            print("\nSaving analysis results...")
            self._save_analysis_results(analysis_results)
            
            print(f"\n[{self.model_backbone}] Analysis complete! Results saved in: {self.results_dir}")

        except Exception as e:
            raise Exception(f"Error during legal analysis: {str(e)}")
