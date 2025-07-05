import transformers
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Cache directory
cache_dir = "../.cache"

# Load the model from the local directory
model = transformers.AutoModelForCausalLM.from_pretrained(cache_dir, torch_dtype=torch.bfloat16)
tokenizer = transformers.AutoTokenizer.from_pretrained(cache_dir)

pipeline = transformers.pipeline(
  "text-generation",
  model=model,
  tokenizer=tokenizer,
  device="cuda",
  pad_token_id=tokenizer.eos_token_id,
  truncation=True,
  max_length=50
)

# # Test the pipeline
# output = pipeline("Will generative AI take over the world?")
# print(output)

# Create FastAPI app
app = FastAPI()

class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 50

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    try:
        result = pipeline(request.prompt, max_length=request.max_length)
        output = result[0]['generated_text']
        print(output)
        return {"generated_text": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}
health_check()