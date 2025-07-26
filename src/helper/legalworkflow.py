import os
import json
import datetime
from typing import Dict, Optional, List
from .agent_clients import AgentClient
from .legalagents import LegalReviewPanel
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
        from .hypothetical_processor import process_hypothetical_directory, split_into_subquestions

        print("\nInitiating legal analysis workflow...")

        if self.hypothetical:
            full_text = process_hypothetical_directory(self.hypothetical, self.hypothetical_indices)
            subquestions = split_into_subquestions(full_text)

            if not self.hypothetical_indices or not all(i < len(subquestions) for i in self.hypothetical_indices):
                raise ValueError("Invalid hypothetical index provided.")

            selected_subqs = [subquestions[i] for i in self.hypothetical_indices]
        else:
            selected_subqs = [self.legal_question]

        # Initialize result dict
        analysis_results = {
            "legal_question": self.legal_question if not self.hypothetical else None,
            "hypothetical": full_text if self.hypothetical else None,
            "timestamp": self.timestamp,
            "model": self.model_backbone,
            "agent_outputs": {},
            "final_synthesis": None
        }

        for i, question in enumerate(selected_subqs):
            print(f"\n🔎 Analyzing Subquestion {self.hypothetical_indices[i] if self.hypothetical else i}: {question}")

            for agent_name, agent in self.agents.items():
                if agent_name not in analysis_results["agent_outputs"]:
                    analysis_results["agent_outputs"][agent_name] = {}

                if f"subquestion_{i+1}" in analysis_results["agent_outputs"][agent_name]:
                    print(f"⏩ Skipping already processed subquestion {i+1} for {agent_name}")
                    continue

                result = agent.perform_full_structured_analysis(question=question)
                analysis_results["agent_outputs"][agent_name][f"subquestion_{i+1}"] = {
                    "question": question,
                    "phased_analysis": result
                }

        # Run synthesis once
        print("\n🧠 Synthesizing perspectives...")
        internal_reviews = [
            entry["phased_analysis"].get("review", "") 
            for entry in analysis_results["agent_outputs"].get("internal", {}).values()
        ]
        external_reviews = [
            entry["phased_analysis"].get("review", "") 
            for entry in analysis_results["agent_outputs"].get("external", {}).values()
        ]

        review_panel = LegalReviewPanel(
            input_model=self.model_backbone,
            api_keys=self.api_keys,
            agent_config=self.agent_configs,
            max_steps=2,
        )
        synthesis = review_panel.synthesize_reviews([
            {"perspective": "internal_law", "review": "\n".join(internal_reviews)},
            {"perspective": "external_law", "review": "\n".join(external_reviews)},
        ], source_text=full_text)

        analysis_results["final_synthesis"] = synthesis

        self._save_analysis_results(analysis_results)
        print(f"\n✅ [{self.model_backbone}] Analysis complete! Results saved in: {self.results_dir}")

    def flatten_agent_results(agent_results: Dict) -> str:
        """Combine all phase results across subquestions into a single string."""
        output = []
        for subq_key, subq_content in agent_results.items():
            subq_text = subq_content.get("subquestion_text", "")
            phase_results = subq_content.get("phase_results", {})
            output.append(f"--- {subq_key}: {subq_text} ---")
            for phase_name, result in phase_results.items():
                output.append(f"[{phase_name.upper()}]\n{result}")
        return "\n\n".join(output)