import argparse
import json
import os
from typing import Dict, Any, Optional
try:
    from .configloader import load_agent_config
except ImportError:
    # for direct script execution if markdown_translator.py is run as a standalone script
    from helper.configloader import load_agent_config

def create_individual_analysis_files(
    results: Dict[Any, Any], 
    base_output_dir: str, 
    model_name: str, 
    hypo_name: str = None, 
    agent_config: dict = {}, 
    is_first_question: bool = True
) -> None:
    """
    Create separate markdown files for internal, external, and final review analysis.
    
    Args:
        results: The analysis results dictionary for ONE model's answer to ONE question
        base_output_dir: Base output directory
        model_name: Name of the model used
        hypo_name: Name of the hypothetical (if applicable)
        is_first_question: Flag to indicate if this is the first question of a hypo
    """
    # Create model-specific directory
    safe_model_name = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    
    if hypo_name:
        # For hypothetical analysis: results/analysis_timestamp/hypothetical_name/model_name/
        model_dir = os.path.join(base_output_dir, hypo_name, safe_model_name)
    else:
        # For direct questions: results/analysis_timestamp/model_name/
        model_dir = os.path.join(base_output_dir, safe_model_name)
    
    os.makedirs(model_dir, exist_ok=True)
    
    # Extract common metadata
    timestamp = results.get('timestamp', 'N/A')
    legal_question = results.get('legal_question')
    hypothetical = results.get('hypothetical')
    
    # just adding this check to be safe but technically its passed on
    if agent_config is None:
        agent_config = load_agent_config()

    # Generate internal analysis file
    if results.get('agent_outputs', {}).get('internal'):
        internal_file = os.path.join(model_dir, 'internal.md')
        _create_markdown('internal', results, internal_file, model_name, timestamp, legal_question, hypothetical, agent_config, is_first_question)
    
    # Generate external analysis file
    if results.get('agent_outputs', {}).get('external'):
        external_file = os.path.join(model_dir, 'external.md')
        _create_markdown('external', results, external_file, model_name, timestamp, legal_question, hypothetical, agent_config, is_first_question)
    
    # Generate review file with synthesis and metrics
    if results.get('final_synthesis'):
        review_file = os.path.join(model_dir, 'review.md')
        _create_markdown('review', results, review_file, model_name, timestamp, legal_question, hypothetical, is_first_question)

# def _create_internal_markdown(results: Dict, output_file: str, model_name: str, timestamp: str, 
#                             legal_question: Optional[str], hypothetical: Optional[str], agent_config:dict, is_first_question: bool = True) -> None:
#     """Create markdown file for internal legal analysis."""
    
#     # For subsequent questions in the same hypo, append to the existing file
#     file_mode = 'w' if is_first_question else 'a'
    
#     markdown = []
    
#     # Header and metadata only for the first question
#     if is_first_question:
#         markdown.append("# Internal Legal Analysis\n")
#         markdown.append("## Metadata\n")
#         markdown.append(f"**Timestamp**: {timestamp}")
#         markdown.append(f"**Model**: {model_name}")
#         markdown.append(f"**Analysis Type**: Internal Legal Perspective\n")
#         markdown.append("\n")
#         markdown.append("## Hypothetical Scenario\n")
#         scenario_part = hypothetical.split("QUESTIONS:")[0].strip() if "QUESTIONS:" in hypothetical else hypothetical
#         markdown.append(scenario_part + "\n")
#     else:
#         markdown.append("\n---\n") # Add a separator for new questions

#     # Question/Hypothetical
#     if legal_question:
#         markdown.append("## Legal Question\n")
#         markdown.append(legal_question + "\n")
#     elif hypothetical:
#         if "QUESTIONS:" in hypothetical:
#             questions_part = hypothetical.split("QUESTIONS:")[1].strip()
#             markdown.append(questions_part + "\n")
    
#     # Internal analysis sections
#     internal_data = results['agent_outputs']['internal']
    
#     # just adding this check to be safe but technically its passed on
#     if agent_config is None:
#         agent_config = load_agent_config()
#     internal_phases = list(agent_config["internal"]["phase_prompts"].keys())

#     for section in internal_phases:
#         if internal_data.get(section).get('model_resp'):
#             section_title = section.replace("_", " ").title()
#             markdown.append(f"## {section_title}\n")
#             markdown.append(internal_data[section]['model_resp'] + "\n")
    
#     # Write file
#     with open(output_file, file_mode, encoding='utf-8') as f:
#         f.write("\n".join(markdown))


# def _create_external_markdown(results: Dict, output_file: str, model_name: str, timestamp: str,
#                             legal_question: Optional[str], hypothetical: Optional[str], agent_config:dict, is_first_question: bool = True) -> None:
#     """Create markdown file for external legal analysis."""
    
#     # For subsequent questions in the same hypo, append to the existing file
#     file_mode = 'w' if is_first_question else 'a'

#     markdown = []
    
#     # Header and metadata only for the first question
#     if is_first_question:
#         markdown.append("# External Legal Analysis\n")
#         markdown.append("## Metadata\n")
#         markdown.append(f"**Timestamp**: {timestamp}")
#         markdown.append(f"**Model**: {model_name}")
#         markdown.append(f"**Analysis Type**: External Legal Perspective\n")
#         markdown.append("\n")
#         markdown.append("## Hypothetical Scenario\n")
#         scenario_part = hypothetical.split("QUESTIONS:")[0].strip() if "QUESTIONS:" in hypothetical else hypothetical
#         markdown.append(scenario_part + "\n")
#     else:
#         markdown.append("\n---\n") # Add a separator for new questions
    
#     # Question/Hypothetical
#     if legal_question:
#         markdown.append("## Legal Question\n")
#         markdown.append(legal_question + "\n")
#     elif hypothetical:
#         if "QUESTIONS:" in hypothetical:
#             questions_part = hypothetical.split("QUESTIONS:")[1].strip()
#             markdown.append(questions_part + "\n")
        
#     # External analysis sections
#     external_data = results['agent_outputs']['external']
    
#     # just adding this check to be safe but technically its passed on
#     if agent_config is None:
#         agent_config = load_agent_config()
#     external_phases = list(agent_config["external"]["phase_prompts"].keys())

#     for section in external_phases:
#         if external_data.get(section).get('model_resp'):
#             section_title = section.replace("_", " ").title()
#             markdown.append(f"## {section_title}\n")
#             markdown.append(external_data[section]['model_resp'] + "\n")
    
#     # Write file
#     with open(output_file, file_mode, encoding='utf-8') as f:
#         f.write("\n".join(markdown))


# def _create_review_markdown(results: Dict, output_file: str, model_name: str, timestamp: str,
#                           legal_question: Optional[str], hypothetical: Optional[str], is_first_question: bool = True) -> None:
#     """Create markdown file for final review with synthesis and metrics."""
    
#     # For subsequent questions in the same hypo, append to the existing file
#     file_mode = 'w' if is_first_question else 'a'

#     markdown = []
    
#     # Header and metadata only for the first question
#     if is_first_question:
#         markdown.append("# Legal Analysis Review\n")
#         markdown.append("## Metadata\n")
#         markdown.append(f"**Timestamp**: {timestamp}")
#         markdown.append(f"**Model**: {model_name}")
#         markdown.append(f"**Analysis Type**: Final Synthesis & Review\n")
#     else:
#         markdown.append("\n---\n") # Add a separator for new questions
    
#     # Question/Hypothetical (brief version)
#     if legal_question:
#         markdown.append("## Legal Question\n")
#         # Truncate if too long for review
#         question_text = legal_question[:500] + "..." if len(legal_question) > 500 else legal_question
#         markdown.append(question_text + "\n")
#     elif hypothetical:
#         if "QUESTIONS:" in hypothetical:
#             questions_part = hypothetical.split("QUESTIONS:")[1].strip()
#             markdown.append(questions_part + "\n")
    
#     # Final synthesis
#     final_synthesis = results.get('final_synthesis', {})
    
#     # Internal perspective summary
#     if final_synthesis.get("internal_perspective"):
#         markdown.append("## Internal Perspective Summary\n")
#         markdown.append(final_synthesis["internal_perspective"] + "\n")
    
#     # External perspective summary
#     if final_synthesis.get("external_perspective"):
#         markdown.append("## External Perspective Summary\n")
#         markdown.append(final_synthesis["external_perspective"] + "\n")
    
#     # Combined synthesis
#     if final_synthesis.get("synthesis"):
#         markdown.append("## Combined Legal Analysis\n")
#         markdown.append(final_synthesis["synthesis"] + "\n")
    
#     # Evaluation metrics
#     if final_synthesis.get("evaluation"):
#         markdown.append("## Evaluation Metrics\n")
        
#         # Scores table
#         if final_synthesis["evaluation"].get("scores"):
#             markdown.append("### Performance Scores\n")
#             markdown.append("| Criteria | Score |")
#             markdown.append("|---------|-------|")
            
#             for criterion, score in final_synthesis["evaluation"]["scores"].items():
#                 criterion_name = criterion.replace("_", " ").title()
#                 markdown.append(f"| {criterion_name} | {score} |")
            
#             # Add average score
#             if final_synthesis["evaluation"].get("average_score"):
#                 markdown.append(f"| **Average** | **{final_synthesis['evaluation']['average_score']}** |\n")
        
#         # Detailed assessments
#         if final_synthesis["evaluation"].get("assessments"):
#             markdown.append("### Detailed Assessment\n")
#             for criterion, assessment in final_synthesis["evaluation"]["assessments"].items():
#                 criterion_name = criterion.replace("_", " ").title()
#                 markdown.append(f"**{criterion_name}**: {assessment}\n")
    
#     # Factual consistency
#     if final_synthesis.get("consistency_evaluation"):
#         markdown.append("## Factual Consistency Analysis\n")
        
#         # Consistency metrics table
#         markdown.append("| Metric | Value |")
#         markdown.append("|-------|-------|")
        
#         # Entailment score
#         if final_synthesis["consistency_evaluation"].get("Entailment Score"):
#             score = final_synthesis["consistency_evaluation"]["Entailment Score"]
#             markdown.append(f"| Entailment Score | {score:.4f} |")
        
#         # Status
#         if "has_factual_inconsistencies" in final_synthesis:
#             status = "Contains inconsistencies" if final_synthesis["has_factual_inconsistencies"] else "Factually consistent"
#             markdown.append(f"| Status | {status} |")
        
#         # Flagged sentences count
#         if final_synthesis["consistency_evaluation"].get("Flagged Sentences"):
#             count = len(final_synthesis["consistency_evaluation"]["Flagged Sentences"])
#             markdown.append(f"| Flagged Sentences | {count} |\n")
        
#         # List flagged sentences
#         if final_synthesis["consistency_evaluation"].get("Flagged Sentences") and len(final_synthesis["consistency_evaluation"]["Flagged Sentences"]) > 0:
#             markdown.append("### Flagged Sentences\n")
#             for i, sentence in enumerate(final_synthesis["consistency_evaluation"]["Flagged Sentences"], 1):
#                 markdown.append(f"{i}. \"{sentence}\"\n")
    
#     # Quality summary
#     markdown.append("## Analysis Quality Summary\n")
    
#     # Create a summary based on available metrics
#     total_score = final_synthesis.get("evaluation", {}).get("average_score")
#     consistency_score = final_synthesis.get("consistency_evaluation", {}).get("Entailment Score")
    
#     if total_score:
#         if total_score >= 7:
#             quality = "Excellent"
#         elif total_score >= 6:
#             quality = "Good"
#         elif total_score >= 5:
#             quality = "Satisfactory"
#         else:
#             quality = "Needs Improvement"
        
#         markdown.append(f"**Overall Quality**: {quality} (Score: {total_score})\n")
    
#     if consistency_score:
#         consistency_status = "High" if consistency_score >= 0.9 else "Moderate" if consistency_score >= 0.7 else "Low"
#         markdown.append(f"**Factual Consistency**: {consistency_status} (Score: {consistency_score:.3f})\n")
    
#     # Write file
#     with open(output_file, file_mode, encoding='utf-8') as f:
#         f.write("\n".join(markdown))


def convert_json_to_individual_md_files(input_file: str, base_output_dir: str = None, model_name: str = None, hypo_name: str = None, agents_config_json_path: str = "../settings/agents.json"):
    """
    Convert a legal analysis JSON file to separate internal, external, and review markdown files.
    
    Args:
        input_file: Path to the JSON file to convert
        base_output_dir: Base output directory
        model_name: Name of the model (extracted from JSON if not provided)
        hypo_name: Name of the hypothetical scenario (if applicable)
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            results = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON file: {input_file}")
        
    if not base_output_dir:
        base_output_dir = os.path.dirname(input_file)

    hypo_name = input_file.split(os.sep)[-1].replace('.json', '') if hypo_name is None else hypo_name
    
    agent_config = load_agent_config(agents_config_json_path)
    
    for i, question_result in enumerate(results.get("results_per_question", [])):
        is_first = (i == 0)
        question_index = question_result.get("question_index", "scenario")
        for model_results in question_result.get("models", []):
            if isinstance(model_results, dict) and "error" not in model_results:
                model_name = model_results.get("model", "unknown_model")
                try:
                    # Use enhanced markdown translator to create individual files
                    create_individual_analysis_files(
                        results=model_results,
                        base_output_dir=base_output_dir,
                        model_name=model_name,
                        hypo_name=hypo_name,
                        agent_config=agent_config,
                        is_first_question=is_first
                    )
                    print(f"Enhanced markdown files created for {model_name} for question {question_index}")
                except Exception as md_error:
                    print(f"Warning: Failed to create enhanced markdown for {model_name} for question {question_index}: {str(md_error)}")
            else:
                model_name = model_results.get("model", "unknown_model")
                print(f"Skipping markdown generation for {model_name} for question {question_index} due to errors in results")
    
    print(f"Created individual analysis files for {model_name} in {base_output_dir}")

# Legacy function to maintain backward compatibility
def convert_to_md(input_file, output_file=None, agents_config_json_path: str = "../settings/agents.json"):
    """
    Legacy function - converts to single markdown file for backward compatibility
    """
    # Generate default output file name if not provided
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + '.md'
    
    # Read JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON file: {input_file}")
    
    agent_config = load_agent_config(agents_config_json_path)

    # Generate markdown content (existing implementation)
    markdown = []
    
    # Add title
    markdown.append("# Legal Analysis Report\n")
    
    # Add metadata
    if any(key in data for key in ["timestamp", "model"]):
        markdown.append("## Metadata\n")
        if data.get("timestamp"):
            markdown.append(f"**Timestamp**: {data['timestamp']}")
        if data.get("model"):
            markdown.append(f"**Model**: {data['model']}\n")
    
    # Process legal question or hypothetical
    if data.get("legal_question"):
        markdown.append("## Legal Question\n")
        markdown.append(data["legal_question"] + "\n")
    elif data.get("hypothetical"):
        markdown.append("## Hypothetical\n")
        # Extract just the questions if they exist
        if "<Questions>" in data["hypothetical"]:
            questions_part = data["hypothetical"].split("<Questions>")[1].strip()
            markdown.append(questions_part + "\n")
        else:
            # Limit length for readability
            hypo_text = data["hypothetical"]
            if len(hypo_text) > 1000:
                hypo_text = hypo_text[:997] + "..."
            markdown.append(hypo_text + "\n")
    
    # Process agent outputs
    if data.get("agent_outputs"):
        markdown.append("## Agent Analysis\n")

        internal_phases = list(agent_config["internal"]["phase_prompts"].keys())
        
        # Process internal agent
        if "internal" in data["agent_outputs"]:
            markdown.append("### Internal Legal Perspective\n")
            
            # Show sections if available
            for section in internal_phases:
                if data["agent_outputs"]["internal"].get(section):
                    section_title = section.capitalize()
                    markdown.append(f"#### {section_title}\n")
                    markdown.append(data["agent_outputs"]["internal"][section]['model_resp'] + "\n")
        
        external_phases = list(agent_config["external"]["phase_prompts"].keys())

        # Process external agent
        if "external" in data["agent_outputs"]:
            markdown.append("### External Legal Perspective\n")
            
            # Show sections if available
            for section in external_phases:
                if data["agent_outputs"]["external"].get(section):
                    section_title = section.capitalize()
                    markdown.append(f"#### {section_title}\n")
                    markdown.append(data["agent_outputs"]["external"][section]['model_resp'] + "\n")
    
    # Process final synthesis
    if data.get("final_synthesis"):
        markdown.append("## Final Synthesis\n")
        
        # Internal perspective
        if data["final_synthesis"].get("internal_perspective"):
            markdown.append("### Internal Perspective Summary\n")
            markdown.append(data["final_synthesis"]["internal_perspective"] + "\n")
        
        # External perspective
        if data["final_synthesis"].get("external_perspective"):
            markdown.append("### External Perspective Summary\n")
            markdown.append(data["final_synthesis"]["external_perspective"] + "\n")
        
        # Combined synthesis
        if data["final_synthesis"].get("synthesis"):
            markdown.append("### Combined Legal Analysis\n")
            markdown.append(data["final_synthesis"]["synthesis"] + "\n")
        
        # Process evaluation scores as a table
        if data["final_synthesis"].get("evaluation") and data["final_synthesis"]["evaluation"].get("scores"):
            markdown.append("### Evaluation Scores\n")
            
            # Create score table
            markdown.append("| Criteria | Score |")
            markdown.append("|---------|-------|")
            
            for criterion, score in data["final_synthesis"]["evaluation"]["scores"].items():
                # Clean up criterion name for better display
                criterion_name = criterion.replace("_", " ").title()
                criterion_name = criterion_name.replace("**", "")  # Remove asterisks if present
                markdown.append(f"| {criterion_name} | {score} |")
            
            # Add average score if available
            if data["final_synthesis"]["evaluation"].get("average_score"):
                markdown.append(f"| **Average** | **{data['final_synthesis']['evaluation']['average_score']}** |\n")
            
            # Add assessment details
            if data["final_synthesis"]["evaluation"].get("assessments"):
                markdown.append("#### Detailed Assessment\n")
                for criterion, assessment in data["final_synthesis"]["evaluation"]["assessments"].items():
                    criterion_name = criterion.replace("_", " ").title().replace("**", "")
                    markdown.append(f"**{criterion_name}**: {assessment}\n")
        
        # Add factual consistency information
        if data["final_synthesis"].get("consistency_evaluation"):
            markdown.append("### Factual Consistency\n")
            
            # Create consistency table
            markdown.append("| Metric | Value |")
            markdown.append("|-------|-------|")
            
            # Add entailment score
            if data["final_synthesis"]["consistency_evaluation"].get("Entailment Score"):
                score = data["final_synthesis"]["consistency_evaluation"]["Entailment Score"]
                markdown.append(f"| Entailment Score | {score:.4f} |")
            
            # Add factual consistency score if different
            if data["final_synthesis"].get("factual_consistency_score") and data["final_synthesis"]["factual_consistency_score"] != score:
                fc_score = data["final_synthesis"]["factual_consistency_score"]
                markdown.append(f"| Factual Consistency Score | {fc_score:.4f} |")
            
            # Add consistency status
            if "has_factual_inconsistencies" in data["final_synthesis"]:
                status = "Contains inconsistencies" if data["final_synthesis"]["has_factual_inconsistencies"] else "Factually consistent"
                markdown.append(f"| Status | {status} |")
            
            # Add number of flagged sentences
            if data["final_synthesis"]["consistency_evaluation"].get("Flagged Sentences"):
                count = len(data["final_synthesis"]["consistency_evaluation"]["Flagged Sentences"])
                markdown.append(f"| Flagged Sentences | {count} |\n")
            
            # Add the flagged sentences
            if data["final_synthesis"]["consistency_evaluation"].get("Flagged Sentences") and len(data["final_synthesis"]["consistency_evaluation"]["Flagged Sentences"]) > 0:
                markdown.append("\n#### Flagged Sentences\n")
                for i, sentence in enumerate(data["final_synthesis"]["consistency_evaluation"]["Flagged Sentences"], 1):
                    markdown.append(f"{i}. \"{sentence}\"\n")
    
    # Write markdown to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(markdown))
    
    print(f"Converted {input_file} to {output_file}")
    return output_file

def _create_markdown(
    analysis_type: str,
    results: Dict,
    output_file: str,
    model_name: str,
    timestamp: str,
    legal_question: Optional[str],
    hypothetical: Optional[str],
    agent_config: dict,
    is_first_question: bool = True
) -> None:
    """
    Create markdown file for internal, external, or review analysis.

    analysis_type: "internal", "external", or "review"
    """
    file_mode = 'w' if is_first_question else 'a'
    markdown = []

    # Header and metadata
    if is_first_question:
        markdown.append(f"# {analysis_type.title()} Analysis\n")
        analysis_label = f"{analysis_type.title()} Perspective"

        markdown.append("## Metadata\n")
        markdown.append(f"**Timestamp**: {timestamp}\n")
        markdown.append(f"**Model**: {model_name}\n")
        markdown.append(f"**Analysis Type**: {analysis_label}\n")
        if analysis_type in ("internal", "external"):
            markdown.append("\n")
            markdown.append("## Hypothetical Scenario\n")
            scenario_part = hypothetical.split("<Questions>")[0].strip() if hypothetical and "<Questions>" in hypothetical else hypothetical
            if scenario_part:
                markdown.append(scenario_part + "\n")
    else:
        markdown.append("\n---\n")  # Separator

    # Question/Hypothetical
    if legal_question:
        markdown.append("## Legal Question\n")
        text = legal_question
        if analysis_type == "review" and len(text) > 500:
            text = text[:500] + "..."
        markdown.append(text + "\n")
    elif hypothetical:
        if "<Questions>" in hypothetical:
            questions_part = hypothetical.split("<Questions>")[1].strip().replace("</Questions>", "")
            markdown.append(questions_part + "\n")

    # Main content
    if analysis_type in ("internal", "external"):
        agent_key = analysis_type
        data = results.get('agent_outputs', {}).get(agent_key, {})
        if agent_config is None:
            agent_config = load_agent_config()
        phases = list(agent_config[agent_key]["phase_prompts"].keys())
        for section in phases:
            if data.get(section) and data[section].get('model_resp'):
                section_title = section.replace("_", " ").title()
                markdown.append(f"## {section_title}\n")
                markdown.append(data[section]['model_resp'] + "\n")
    elif analysis_type == "review":
        final_synthesis = results.get('final_synthesis', {})
        # Internal perspective summary
        if final_synthesis.get("internal_perspective"):
            markdown.append("## Internal Perspective Summary\n")
            markdown.append(final_synthesis["internal_perspective"] + "\n")
        # External perspective summary
        if final_synthesis.get("external_perspective"):
            markdown.append("## External Perspective Summary\n")
            markdown.append(final_synthesis["external_perspective"] + "\n")
        # Combined synthesis
        if final_synthesis.get("synthesis"):
            markdown.append("## Combined Legal Analysis\n")
            markdown.append(final_synthesis["synthesis"] + "\n")
        # Evaluation metrics
        if final_synthesis.get("evaluation"):
            markdown.append("## Evaluation Metrics\n")
            if final_synthesis["evaluation"].get("scores"):
                markdown.append("### Performance Scores\n")
                markdown.append("| Criteria | Score |")
                markdown.append("|---------|-------|")
                for criterion, score in final_synthesis["evaluation"]["scores"].items():
                    criterion_name = criterion.replace("_", " ").title()
                    markdown.append(f"| {criterion_name} | {score} |")
                if final_synthesis["evaluation"].get("average_score"):
                    markdown.append(f"| **Average** | **{final_synthesis['evaluation']['average_score']}** |\n")
            if final_synthesis["evaluation"].get("assessments"):
                markdown.append("### Detailed Assessment\n")
                for criterion, assessment in final_synthesis["evaluation"]["assessments"].items():
                    criterion_name = criterion.replace("_", " ").title()
                    markdown.append(f"**{criterion_name}**: {assessment}\n")
        # Factual consistency
        if final_synthesis.get("consistency_evaluation"):
            markdown.append("## Factual Consistency Analysis\n")
            markdown.append("| Metric | Value |")
            markdown.append("|-------|-------|")
            if final_synthesis["consistency_evaluation"].get("Entailment Score"):
                score = final_synthesis["consistency_evaluation"]["Entailment Score"]
                markdown.append(f"| Entailment Score | {score:.4f} |")
            if "has_factual_inconsistencies" in final_synthesis:
                status = "Contains inconsistencies" if final_synthesis["has_factual_inconsistencies"] else "Factually consistent"
                markdown.append(f"| Status | {status} |")
            if final_synthesis["consistency_evaluation"].get("Flagged Sentences"):
                count = len(final_synthesis["consistency_evaluation"]["Flagged Sentences"])
                markdown.append(f"| Flagged Sentences | {count} |\n")
            if final_synthesis["consistency_evaluation"].get("Flagged Sentences") and len(final_synthesis["consistency_evaluation"]["Flagged Sentences"]) > 0:
                markdown.append("### Flagged Sentences\n")
                for i, sentence in enumerate(final_synthesis["consistency_evaluation"]["Flagged Sentences"], 1):
                    markdown.append(f"{i}. \"{sentence}\"\n")
        # Quality summary
        markdown.append("## Analysis Quality Summary\n")
        total_score = final_synthesis.get("evaluation", {}).get("average_score")
        consistency_score = final_synthesis.get("consistency_evaluation", {}).get("Entailment Score")
        if total_score:
            if total_score >= 7:
                quality = "Excellent"
            elif total_score >= 6:
                quality = "Good"
            elif total_score >= 5:
                quality = "Satisfactory"
            else:
                quality = "Needs Improvement"
            markdown.append(f"**Overall Quality**: {quality} (Score: {total_score})\n")
        if consistency_score:
            consistency_status = "High" if consistency_score >= 0.9 else "Moderate" if consistency_score >= 0.7 else "Low"
            markdown.append(f"**Factual Consistency**: {consistency_status} (Score: {consistency_score:.3f})\n")
    print(markdown, output_file)
    # Write file
    with open(output_file, file_mode, encoding='utf-8') as f:
        f.write("\n".join(markdown))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Result JSON to Markdown Converter")
    parser.add_argument("--inf", type=str, required=True, help="Path to results.json file")
    parser.add_argument("--outf", type=str, default=None, help="Path to output markdown file")
    parser.add_argument("--agentsf", type=str, default=None, help="Path to agents.json file")
    args = parser.parse_args()
    convert_json_to_individual_md_files(args.inf, args.outf, agents_config_json_path=args.agentsf)

