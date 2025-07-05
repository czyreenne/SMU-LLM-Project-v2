import json
import os
import re
import logging
from typing import Dict, List, Union

import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def identify_origin(model_name):
    google_pattern = r"\b(gemini|palm|imagen|codey|veo|medlm|learnlm|seclm)\b"
    openai_pattern = r"\b(gpt|dall|tts|codex|clip)\b"
    anthropic_pattern = r"\b(claude)\b"

    # Check the model name against each pattern
    if re.search(google_pattern, model_name, re.IGNORECASE):
        return "google"
    elif re.search(openai_pattern, model_name, re.IGNORECASE):
        return "openai"
    elif re.search(anthropic_pattern, model_name, re.IGNORECASE):
        return "anthropic"
    else:
        return "others"


class ModelPredictor:
    def __init__(self, system_prompt, model_name):
        self.system_prompt = system_prompt
        self.max_length = 2000
        self.model_name = model_name
        
        # Load settings from settings.json
        with open('settings.json', 'r') as settings_file:
            settings = json.load(settings_file)
        
        self.origin = identify_origin(model_name)
        if self.origin == 'others':
            # Load settings of each pipeline below:
            self.pipeline = settings["pipeline"]
            if self.pipeline == "vertex_ai":
                from google.cloud import aiplatform
                """
                Make sure to run the following commands before running the code: 
                1. pip install google-cloud-aiplatform transformers torch fastapi pydantic
                2. gcloud auth application-default login
                """
                self.project = "775367805714"
                self.endpoint_id = "7326375829359820800"
                self.location = "asia-southeast1"
                self.api_endpoint = "asia-southeast1-aiplatform.googleapis.com"
                self.client_options = {"api_endpoint": self.api_endpoint}
                self.client = aiplatform.gapic.PredictionServiceClient(client_options=self.client_options)
                self.endpoint = self.client.endpoint_path(project=self.project, location=self.location, endpoint=self.endpoint_id)
            
            elif self.pipeline == "local_model":
                import transformers
                import torch

                cache_dir = f"../.cache/{model_name}"
                self.local_model = transformers.AutoModelForCausalLM.from_pretrained(cache_dir, torch_dtype=torch.bfloat16)
                self.local_tokenizer = transformers.AutoTokenizer.from_pretrained(cache_dir)
                self.local_pipeline = transformers.pipeline(
                    "text-generation",
                    model=self.local_model,
                    tokenizer=self.local_tokenizer,
                    device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
                    pad_token_id=self.local_tokenizer.eos_token_id,
                    truncation=True,
                    max_length=self.max_length
                )


        
    def predict(self, user_prompt: str = None) -> str:
        prompt = f"{self.system_prompt} \n{user_prompt}\n"
        if self.origin == 'others':
            if self.pipeline == "vertex_ai":
                from google.protobuf import json_format
                from google.protobuf.struct_pb2 import Value

                instances = [{"prompt": prompt, "max_tokens": self.max_length}]
                formatted_instances = [json_format.ParseDict(instance_dict, Value()) for instance_dict in instances]
                parameters_dict = {}
                parameters = json_format.ParseDict(parameters_dict, Value())

                try:
                    response = self.client.predict(endpoint=self.endpoint, instances=formatted_instances, parameters=parameters)
                    predictions = response.predictions
                    cleaned_predictions = []
                    for prediction in predictions:
                        parts = prediction.split("My answer is:")
                        cleaned_text = parts[-1] if len(parts) > 1 else prediction
                        cleaned_predictions.append(cleaned_text)
                    return cleaned_predictions[0]
                except Exception as e:
                    logger.error(f"Prediction failed: {e}")
                    return f"Prediction failed: {e}"

            elif self.pipeline == "local_model":
                try:
                    predictions = self.local_pipeline(prompt)
                    return predictions[0]["generated_text"]
                except Exception as e:
                    logger.error(f"Local model prediction failed: {e}")
                    return f"Local model prediction failed: {e}"

            elif self.pipeline == "lm_studio": 
                try:            
                    data = {
                        "messages": [
                            { "role": "system", "content": self.system_prompt },
                            { "role": "user", "content": user_prompt }
                        ],
                        "temperature": 0.2,
                        "max_tokens": 600,
                        "stream": False
                        }
                    
                    # Define the API URL and headers
                    url = "http://localhost:1234/v1/chat/completions"
                    headers = {
                        "Content-Type": "application/json"
                    }

                    # Send the POST request
                    response = requests.post(url, headers=headers, data=json.dumps(data))
                    response_data = response.json()

                    # Handle the response
                    if response.ok:
                        return response_data['choices'][0]['message']['content']                         
                
                except Exception as e:
                    logger.error(f"LM Studio prediction failed: {e}")
                    return f"LM Studio prediction failed: {e}"

        elif self.origin == "openai":
            from openai import OpenAI

            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            try:
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=self.messages + [{"role": "user", "content": user_prompt}]
                )
                # Extracting the text content from the response
                return completion.choices[0].message.content
            except Exception as e:
                logger.error(f"OpenAI model prediction failed: {e}")
                return f"OpenAI model prediction failed: {e}"