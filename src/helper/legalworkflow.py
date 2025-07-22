import os
import json
import datetime
from typing import Dict, Optional, List
from .agent_clients import AgentClient
from .configloader import load_agent_config
from .markdown_translator import create_individual_analysis_files


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
        # Import here to avoid circular imports
        from .hypothetical_processor import process_hypothetical_directory
        print(self.agents)
        try:
            print("\nInitiating legal analysis workflow...")

            if self.hypothetical:
                analysis_text = process_hypothetical_directory(self.hypothetical, self.hypothetical_indices)
                print(f"\nLegal question: {analysis_text}")
                analysis_results = {
                    "legal_question": None,
                    "hypothetical": analysis_text,
                    "timestamp": self.timestamp,
                    "model": self.model_backbone,
                    "agent_outputs": {},
                    "final_synthesis": None
                }
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

            # Save all results
            print("\nSaving analysis results...")
            self._save_analysis_results(analysis_results)
            
            print(f"\n[{self.model_backbone}] Analysis complete! Results saved in: {self.results_dir}")

        except Exception as e:
            raise Exception(f"Error during legal analysis: {str(e)}")
