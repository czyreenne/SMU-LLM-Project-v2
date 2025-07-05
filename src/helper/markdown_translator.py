import json
import os

def convert_to_md(input_file, output_file=None):
    """
    Convert a legal analysis JSON file to a formatted Markdown file.
    
    Args:
        input_file: Path to the JSON file to convert
        output_file: Path to save the Markdown file (if None, uses input_file with .md extension)
        
    Returns:
        str: Path to the created Markdown file
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
    
    # Generate markdown content
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
        if "QUESTIONS:" in data["hypothetical"]:
            questions_part = data["hypothetical"].split("QUESTIONS:")[1].strip()
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
        
        # Process internal agent
        if "internal" in data["agent_outputs"]:
            markdown.append("### Internal Legal Perspective\n")
            
            # Show sections if available
            for section in ["issues", "rules", "analysis", "conclusion"]:
                if data["agent_outputs"]["internal"].get(section):
                    section_title = section.capitalize()
                    markdown.append(f"#### {section_title}\n")
                    markdown.append(data["agent_outputs"]["internal"][section] + "\n")
        
        # Process external agent
        if "external" in data["agent_outputs"]:
            markdown.append("### External Legal Perspective\n")
            
            # Show sections if available
            for section in ["issues", "rules", "analysis", "conclusion"]:
                if data["agent_outputs"]["external"].get(section):
                    section_title = section.capitalize()
                    markdown.append(f"#### {section_title}\n")
                    markdown.append(data["agent_outputs"]["external"][section] + "\n")
    
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

def batch_convert_directory(directory_path, output_directory=None):
    """
    Convert all JSON files in a directory to Markdown files.
    
    Args:
        directory_path: Path to directory containing JSON files
        output_directory: Directory to save Markdown files (if None, uses same directory)
    
    Returns:
        list: Paths to all created Markdown files
    """
    if output_directory is None:
        output_directory = directory_path
    else:
        os.makedirs(output_directory, exist_ok=True)
    
    converted_files = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            input_path = os.path.join(directory_path, filename)
            output_path = os.path.join(output_directory, os.path.splitext(filename)[0] + '.md')
            
            try:
                converted_file = convert_to_md(input_path, output_path)
                converted_files.append(converted_file)
            except Exception as e:
                print(f"Error converting {filename}: {e}")
    
    return converted_files