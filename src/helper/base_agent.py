import time
from typing import Any, Dict, List, Optional

from helper.inference import get_provider, query_model


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
