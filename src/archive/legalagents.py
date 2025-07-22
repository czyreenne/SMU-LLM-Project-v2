from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import json
import time
from helper.inference import *
from helper.eval import SummaryEvaluator

@dataclass
class ReviewCriteria:
    """Data class for review criteria scores"""
    jurisdictional_understanding: int
    legal_reasoning: int
    comparative_analysis: int
    practical_application: int
    academic_merit: int

    def validate(self) -> bool:
        """Validate all scores are within acceptable range"""
        return all(
            1 <= getattr(self, field.name) <= 10
            for field in self.__dataclass_fields__.values()
        )

class BaseAgent:
    """Base class for all legal analysis agents"""
    
    def __init__(
        self, 
        agent_type: str,
        input_model: str,
        config: Dict[str, Any],
        api_keys: Dict[str, str],
        notes: Optional[List[Dict[str, Any]]] = None
        
    ):
        """
        Initialize base agent
        
        Args:
            agent_type: Type of agent (internal, external etc)
            input_model: Model to use for this agent
            config: Configuration dictionary loaded from JSON
            api_keys: Dictionary of API keys by provider
            notes: List of notes/instructions for the agent
        """
        if agent_type not in config:
            raise ValueError(f"Invalid agent type: {agent_type}")
            
        agent_config = config[agent_type]
        default_config = agent_config['default_config']
        
        self.agent_type = agent_type
        self.role_desc = agent_config['role_description']
        self.phase_prompts = agent_config['phase_prompts']
        self.phases = list(self.phase_prompts.keys())
        self.notes = notes or []
        self.max_steps = default_config['max_steps']
        self.model = input_model or default_config['model']
        self.history: List[tuple[Optional[int], str]] = []
        self.prev_comm = ""
        self.api_keys = api_keys or {}
        self.max_hist_len = default_config['max_history']
        
        # Get the appropriate API key based on the model
        provider = get_provider(self.model)
        self.api_key = self.api_keys.get(provider)
        
        # Rate limiting
        self.last_api_call = 0
        self.min_api_interval = 1  # seconds

    def _rate_limit(self) -> None:
        """Implement rate limiting for API calls"""
        now = time.time()
        time_since_last = now - self.last_api_call
        if time_since_last < self.min_api_interval:
            time.sleep(self.min_api_interval - time_since_last)
        self.last_api_call = now

    def _manage_history(self, entry: str) -> None:
        """Manage history entries with cleanup"""
        self.history.append((None, entry))
        while len(self.history) > self.max_hist_len:
            self.history.pop(0)

    def role_description(self) -> str:
        """Return the role description from config"""
        return self.role_desc

    def phase_prompt(self, phase: str) -> str:
        """Return the phase prompt from config"""
        if phase not in self.phase_prompts:
            raise ValueError(f"Invalid phase {phase} for agent {self.agent_type}")
        return self.phase_prompts[phase]

    def inference(
        self,
        question: str,
        phase: str,
        step: int,
        feedback: str = "",
        temp: Optional[float] = None
    ) -> str:
        """
        Args:
            question: The legal question to analyse
            phase: Current phase of analysis
            step: Current step number
            feedback: Previous feedback
            temp: Temperature for model inference
            
        Returns:
            Model response
        """
        if phase not in self.phases:
            raise ValueError(f"Invalid phase {phase} for agent {self.__class__.__name__}")
            
        self._rate_limit()
        
        system_prompt = (
            f"You are {self.role_description()}\n"
            f"Task instructions: {self.phase_prompt(phase)}\n"
        )
        
        history_str = "\n".join(entry[1] for entry in self.history)
        phase_notes = [
            note["note"] for note in self.notes 
            if phase in note["phases"]
        ]
        notes_str = (
            f"Notes for the task objective: {phase_notes}\n" 
            if phase_notes else ""
        )
        
        user_prompt = (
            f"History: {history_str}\n{'~' * 10}\n"
            f"Current Step #{step}, Phase: {phase}\n"
            f"[Objective] Your goal is to analyse the following legal question: "
            f"{question}\n"
            f"Feedback: {feedback}\nNotes: {notes_str}\n"
            f"Your previous response was: {self.prev_comm}. "
            f"Please ensure your new analysis adds value.\n"
            f"Please provide your analysis below:\n"
        )

        try:
            model_resp = query_model(
                model_str=self.model,
                system_prompt=system_prompt,
                prompt=user_prompt,
                api_key=self.api_key,
                temp=temp
            )
        except Exception as e:
            print(f"Error during model inference: {str(e)}")
            raise
            
        self.prev_comm = model_resp
        self._manage_history(
            f"Step #{step}, Phase: {phase}, Analysis: {model_resp}"
        )
        
        return model_resp

class Internal(BaseAgent):
    def __init__(
        self,
        input_model: str,
        api_keys: Dict[str, str],
        config: Dict[str, Any],
        notes: Optional[List[Dict[str, Any]]] = None
    ):
        super().__init__(
            agent_type='internal',
            input_model = input_model,
            config=config,
            api_keys=api_keys,
            notes=notes
            
        )
        
        self.perspective = "internal_law"
        self.sg_statutes = {}                           #TODO implement VDB for contextual knowledge
        self.sg_case_law = {}

class External(BaseAgent):
    def __init__(
        self,
        input_model: str,
        api_keys: Dict[str, str],
        config: Dict[str, Any],
        notes: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(
            agent_type='external',
            input_model = input_model,
            config=config,
            api_keys=api_keys,
            notes=notes
        )
        
        self.perspective = "external_law"
        self.us_constitution = {}                           #TODO implement VDB for contextual knowledge
        self.us_case_law = {}

class LegalReviewPanel:
    """Specialized review panel for Singapore legal analysis with factual consistency checking"""
    
    def __init__(
        self,
        agent_config: Dict[str, Any],
        input_model: str,
        api_keys: Dict[str, str],
        max_steps: int = 100,
        max_history: int = 15,
        notes: Optional[List[Dict[str, Any]]] = None,
        review_config_path: str = "settings/review.json",
        entailment_threshold: float = 0.84
    ):
        """
        Initialize the review panel with configuration
        
        Args:
            agent_config: Configuration for agents
            input_model: Model to use for synthesis
            api_keys: Dictionary of API keys by provider
            max_steps: Maximum steps for agent analysis
            max_history: Maximum history entries to maintain
            notes: Additional notes or instructions for agents
            review_config_path: Path to review configuration file
            entailment_threshold: Threshold for determining factual consistency
        """
        self.model = input_model if input_model is not None else "gpt-4o-mini"
        provider = get_provider(self.model)
        self.api_key = api_keys.get(provider)
        
        # Load configurations
        try:
            with open(review_config_path, 'r') as f:
                self.review_config = json.load(f)
        except Exception as e:
            raise Exception(f"Error loading configurations: {str(e)}")
        
        # Initialize reviewers with configuration
        self.reviewers = [
            Internal(
                input_model=input_model,
                api_keys=api_keys,
                config=agent_config,
                notes=notes,
            ),
            External(
                input_model=input_model,
                api_keys=api_keys,
                config=agent_config,
                notes=notes,
            )
        ]
        
        # Initialize factual consistency evaluator
        self.consistency_evaluator = SummaryEvaluator(entailment_threshold=entailment_threshold)

    def _query_model(self, system_prompt: str, prompt: str) -> str:
        """Helper method to safely query the model with error handling"""
        try:
            return query_model(
                model_str=self.model,
                system_prompt=system_prompt,
                prompt=prompt,
                api_key=self.api_key
            )
        except Exception as e:
            print(f"Error querying model: {str(e)}")
            raise

    def evaluate_legal_analysis(self, legal_text: str, source_text: str) -> Dict[str, Any]:
        """
        Evaluate a legal text based on the criteria defined in review_config
        
        Args:
            legal_text: The legal analysis to evaluate
            
        Returns:
            Dictionary containing scores and qualitative assessments
        """
        # Initialize result structure
        evaluation_result = {
            "scores": {},
            "assessments": {},
            "average_score": 0,
            "overall_assessment": ""
        }
        
        # Prepare evaluation prompt
        sys_prompt = (
            "You are an expert legal evaluation system. "
            "Assess the provided legal analysis for a hypothethical scenario or a legal question based on the specified criteria. "
            "Provide numerical scores (1-10) and detailed qualitative assessments for each criterion."
        )
        
        eval_prompt = "Evaluate the following legal analysis according to these criteria:\n\n"
        
        # Add criteria descriptions
        for criterion, details in self.review_config["review_criteria"].items():
            eval_prompt += f"## {criterion.replace('_', ' ').title()}\n"
            eval_prompt += f"Description: {details['description']}\n"
            eval_prompt += "Scoring guide:\n"
            for score_range, description in details["scoring_guide"].items():
                eval_prompt += f"- {score_range}: {description}\n"
            eval_prompt += "\n"
        
        eval_prompt += f"Hypothethical senario/question:\n\n{legal_text}\n\n"
        eval_prompt += f"Legal Analysis to Evaluate:\n\n{source_text}\n\n"
        eval_prompt += "For each criterion, provide:\n"
        eval_prompt += "1. A numerical score between 1-10\n"
        eval_prompt += "2. A detailed qualitative assessment explaining the score\n\n"
        eval_prompt += "Format your response as follows:\n"
        eval_prompt += "Criterion: [criterion name]\nScore: [numerical score 1-10]\nAssessment: [detailed assessment]\n\n"
        eval_prompt += "End with an Overall Assessment summarizing strengths and weaknesses of the analysis."
        
        try:
            # Query model for evaluation
            evaluation_response = self._query_model(sys_prompt, eval_prompt)
            
            # Parse the evaluation response
            sections = evaluation_response.split("Criterion:")
            
            # Process each criterion evaluation
            for section in sections[1:]:  # Skip the first empty split
                lines = section.strip().split("\n")
                if len(lines) >= 3:
                    criterion_name = lines[0].strip().lower().replace(" ", "_")
                    score_line = lines[1].strip()
                    
                    # Fix: Improve score extraction to handle non-standard formats
                    if ":" in score_line:
                        score_text = score_line.split(":")[1].strip()
                        # Remove any non-numeric characters and convert to int
                        score_digits = ''.join(c for c in score_text if c.isdigit())
                        score = int(score_digits) if score_digits else 0
                    else:
                        score = 0
                        
                    # Extract assessment (may be multiple lines)
                    assessment_start = 0
                    for i, line in enumerate(lines[2:], 2):
                        if line.strip().startswith("Assessment:"):
                            assessment_start = i
                            break
                    
                    # Gather assessment text until next criterion or overall assessment
                    assessment_lines = []
                    for i in range(assessment_start, len(lines)):
                        line = lines[i].strip()
                        if (line.startswith("Criterion:") or 
                            line.startswith("Overall Assessment:") or
                            not line):
                            break
                        if line.startswith("Assessment:"):
                            assessment_lines.append(line[len("Assessment:"):].strip())
                        else:
                            assessment_lines.append(line)
                    
                    assessment = " ".join(assessment_lines).strip()
                    
                    # Store in results
                    evaluation_result["scores"][criterion_name] = score
                    evaluation_result["assessments"][criterion_name] = assessment
            
            # Extract overall assessment
            if "Overall Assessment:" in evaluation_response:
                overall_section = evaluation_response.split("Overall Assessment:")[1].strip()
                evaluation_result["overall_assessment"] = overall_section
            
            # Calculate average score
            if evaluation_result["scores"]:
                evaluation_result["average_score"] = round(
                    sum(evaluation_result["scores"].values()) / len(evaluation_result["scores"]), 2
                )
            
            return evaluation_result
            
        except Exception as e:
            print(f"Error in evaluation: {str(e)}")
            # Fallback scoring
            for criterion in self.review_config["review_criteria"].keys():
                evaluation_result["scores"][criterion] = 5
                evaluation_result["assessments"][criterion] = "Evaluation failed"
            evaluation_result["average_score"] = 5
            evaluation_result["overall_assessment"] = f"Automated evaluation failed: {str(e)}"
            return evaluation_result
                    
    def evaluate_factual_consistency(self, source_text: str, generated_text: str) -> Dict[str, Any]:
        """
        Evaluate factual consistency between source text and generated analysis
        
        Args:
            source_text: Original source document
            generated_text: Generated legal analysis
            
        Returns:
            Dictionary with entailment scores and flagged sentences
        """
        try:
            return self.consistency_evaluator.evaluate_summary(source_text, generated_text)
        except Exception as e:
            print(f"Error in factual consistency evaluation: {str(e)}")
            return {
                "Entailment Score": 0.0,
                "Sentence Scores": {},
                "Flagged Sentences": ["Factual consistency evaluation failed"],
                "Error": str(e)
            }
    
    def synthesize_reviews(self, reviews: List[Dict[str, Any]], source_text: str = None) -> Dict[str, Any]:
        """
        Synthesize reviews with Singapore focus and provide evaluation
        
        Args:
            reviews: List of review dictionaries with perspective and content
            source_text: Original source document to check consistency against (optional)
            
        Returns:
            Dictionary containing synthesized analysis, evaluation, and consistency check
        """
        # Validate required perspectives
        required_perspectives = {"internal_law", "external_law"}
        provided_perspectives = {r["perspective"] for r in reviews}
        print('check 2')
        
        if not required_perspectives.issubset(provided_perspectives):
            missing = required_perspectives - provided_perspectives
            raise ValueError(f"Missing required perspectives: {missing}")
        print('check 3')
        # Extract perspectives
        internal_perspective = next(r["review"] for r in reviews if r["perspective"] == "internal_law")
        external_perspective = next(r["review"] for r in reviews if r["perspective"] == "external_law")
        
        # Prepare synthesis
        sys_prompt = self.review_config["synthesis"]["system_prompt"]
        synthesis_prompt = self.review_config["synthesis"]["synthesis_template"].format(
            internal_perspective=internal_perspective,
            external_perspective=external_perspective
        )
        
        try:
            print('check 4')
            # Generate synthesis
            synthesis_text = self._query_model(sys_prompt, synthesis_prompt)
            
            # Evaluate the synthesis
            print('check 5')
            evaluation = self.evaluate_legal_analysis(synthesis_text, source_text)
            
            # Result structure
            result = {
                "internal_perspective": internal_perspective,
                "external_perspective": external_perspective,
                "synthesis": synthesis_text,
                "evaluation": evaluation
            }
            
            # Add factual consistency check if source text is provided
            if source_text:
                print('check 6')
                consistency_evaluation = self.evaluate_factual_consistency(source_text, synthesis_text)
                result["consistency_evaluation"] = consistency_evaluation
                print('check 7')
                # Add a warning flag if factual inconsistencies are detected
                if consistency_evaluation["Flagged Sentences"]:
                    result["has_factual_inconsistencies"] = True
                    result["factual_consistency_score"] = consistency_evaluation["Entailment Score"]
                else:
                    result["has_factual_inconsistencies"] = False
                    result["factual_consistency_score"] = consistency_evaluation["Entailment Score"]
            
            return result
            
        except Exception as e:
            print(f"Error in synthesis: {str(e)}")
            raise
            
    def get_reviewer(self, perspective: str) -> Optional[Any]:
        """Get reviewer by perspective"""
        for reviewer in self.reviewers:
            if reviewer.perspective == perspective:
                return reviewer
        return None