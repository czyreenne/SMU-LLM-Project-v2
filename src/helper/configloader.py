from typing import Dict, Any
import json

def load_agent_config(config_path: str = "settings/agents.json") -> Dict[str, Any]:
    """
    Load and validate agent configuration from JSON file
    
    Args:
        config_path: Path to the configuration JSON file
        
    Returns:
        Dictionary containing validated agent configurations
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Validate required agent types
        required_agents = {'internal', 'external'}
        if not all(agent in config for agent in required_agents):
            missing = required_agents - set(config.keys())
            raise ValueError(f"Missing required agent configurations: {missing}")
            
        # Validate structure for each agent
        for agent_type, agent_config in config.items():
            required_fields = {
                'role_description', 
                'phase_prompts', 
                'default_config'
            }
            if not all(field in agent_config for field in required_fields):
                missing = required_fields - set(agent_config.keys())
                raise ValueError(
                    f"Missing required fields for {agent_type}: {missing}"
                )
            
            # Validate phase prompts
            if not isinstance(agent_config['phase_prompts'], dict):
                raise ValueError(
                    f"Phase prompts for {agent_type} must be a dictionary"
                )
            
            # Validate default config
            required_config = {'model', 'max_steps', 'max_history'}
            if not all(
                field in agent_config['default_config'] 
                for field in required_config
            ):
                missing = required_config - set(agent_config['default_config'].keys())
                raise ValueError(
                    f"Missing required default config for {agent_type}: {missing}"
                )
                
        return config
        
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}"
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {str(e)}")
        
def save_agent_config(
    config: Dict[str, Any], 
    config_path: str = "agents.json"
) -> None:
    """
    Save agent configuration to JSON file
    
    Args:
        config: Dictionary containing agent configurations
        config_path: Path to save the configuration JSON file
    """
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        raise Exception(f"Error saving configuration: {str(e)}")

