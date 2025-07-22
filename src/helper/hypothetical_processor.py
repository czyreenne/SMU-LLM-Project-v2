import os
import sys
import json
import datetime
import subprocess
import multiprocessing
from typing import Dict, Optional, List
from .agent_clients import AgentClient
from ..archive.legalagents import LegalReviewPanel
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
            subprocess.run([sys.executable, "helper/extract_hypo.py", "--inpath", hypo_dir, "--outpath", processed_dir], check=True)
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
        for question_result in hypo_results.get("results_per_question", []):
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
                            hypo_name=f"{hypo_filename}_q{question_index}"
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
        analysis_text, hypo_filename, hypo_original_name = process_single_hypothetical(hypo_dir, hypo_index)
        
        print(f"\n=== ANALYZING HYPOTHETICAL {hypo_index}: {hypo_original_name} ===")
        
        # Dictionary to store results from all models for this hypothetical
        hypo_results = {
            "hypothetical_info": {
                "original_name": hypo_original_name,
                "index": hypo_index,
                "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            },
            "results": []
        }
        
        # Run all models for this hypothetical in parallel
        print(f"Running {len(models)} models in parallel...")
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
        
        # Wait for all models to complete for this hypothetical
        completed = 0
        for i, p in enumerate(processes):
            p.join()
            completed += 1
            print(f"Completed {completed}/{len(models)} models")
        
        # Collect results into a list
        results_list = []
        for model in models:
            if model in results_dict:
                results_list.append(dict(results_dict[model]))
            else:
                results_list.append({"model": model, "error": "Analysis failed", "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")})
        
        hypo_results["results"] = results_list
        
        # Save results for this hypothetical
        print("Saving results...")
        save_individual_hypothetical_results(hypo_results, hypo_filename, shared_results_dir)
        
        print(f"✓ Completed analysis for hypothetical {hypo_index}: {hypo_original_name}")
        
    except Exception as e:
        print(f"✗ Failed to analyze hypothetical {hypo_index}: {str(e)}")


def run_single_model_for_hypothetical(model: str, analysis_text: str, api_keys: dict, results_dict):
    """
    Run a single model analysis for a hypothetical and store result
    """
    try:
        # Import here to avoid circular imports
        from .legalworkflow import LegalSimulationWorkflow
        
        # Create workflow without saving individual files
        workflow = LegalSimulationWorkflow(
            legal_question="",
            api_keys=api_keys,
            model_backbone=model,
            hypothetical="",  # We're passing the text directly
            hypothetical_indices=None,
            shared_results_dir=None  # Don't save individual files
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
        for i, (agent_name, agent) in enumerate(workflow.agents.items()):
            agent_results = agent.perform_full_structured_analysis(question=analysis_text)
            print(f"[{model}] Analysis completed for agent: {agent_name} (Step {i+1}/{len(workflow.agents)})")
            analysis_results["agent_outputs"][agent_name] = agent_results

        # Synthesize reviews
        internal_review = analysis_results["agent_outputs"]["internal"].get("review", "")
        external_review = analysis_results["agent_outputs"]["external"].get("review", "")

        reviews = [
            {"perspective": "internal_law", "review": internal_review},
            {"perspective": "external_law", "review": external_review}
        ]
        
        review_panel = LegalReviewPanel(
            input_model=model,
            api_keys=api_keys,
            agent_config=workflow.agent_configs,
            max_steps=len(reviews),
        )
        synthesis = review_panel.synthesize_reviews(reviews, source_text=analysis_text)
        analysis_results["final_synthesis"] = synthesis
        
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
        # Import here to avoid circular imports
        from .legalworkflow import LegalSimulationWorkflow
        
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
