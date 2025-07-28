import time, tiktoken
from openai import OpenAI
import openai
import os, anthropic, json
from dotenv import dotenv_values

config = dotenv_values("../.env")

TOKENS_IN = dict()
TOKENS_OUT = dict()

encoding = tiktoken.get_encoding("cl100k_base")

def get_provider(model):
    provider_map = {
        "gpt-4o": "openai",
        "gpt-4o-mini": "openai",
        "o1-preview": "openai",
        "o1-mini": "openai",
        "claude-3-5-sonnet": "anthropic",
        "deepseek-chat": "deepseek",
    }
    return provider_map.get(model, "unknown")

def curr_cost_est():
    costmap_in = {
        "gpt-4o": 2.50 / 1000000,
        "gpt-4o-mini": 0.150 / 1000000,
        "o1-preview": 15.00 / 1000000,
        "o1-mini": 3.00 / 1000000,
        "claude-3-5-sonnet": 3.00 / 1000000,
        "deepseek-chat": 1.00 / 1000000,
    }
    costmap_out = {
        "gpt-4o": 10.00/ 1000000,
        "gpt-4o-mini": 0.6 / 1000000,
        "o1-preview": 60.00 / 1000000,
        "o1-mini": 12.00 / 1000000,
        "claude-3-5-sonnet": 12.00 / 1000000,
        "deepseek-chat": 5.00 / 1000000,
    }
    return sum([costmap_in[_]*TOKENS_IN[_] for _ in TOKENS_IN]) + sum([costmap_out[_]*TOKENS_OUT[_] for _ in TOKENS_OUT])

def query_model(model_str, prompt, system_prompt, api_key, tries=5, timeout=5.0, temp=None, print_cost=True, version="1.5"):

    provider = get_provider(model_str)
    
    # Set the API key for the appropriate provider
    if provider == "openai":
        openai.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key or config["OPENAI_API_KEY"]
    elif provider == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = api_key or config["ANTHROPIC_API_KEY"]
    elif provider == "deepseek":
        os.environ["DEEPSEEK_API_KEY"] = api_key or config["DEEPSEEK_API_KEY"]
    else:
        raise Exception(f"Unknown provider for model: {model_str}")
    
    for _ in range(tries):
        try:
            if model_str == "gpt-4o-mini" or model_str == "gpt4omini" or model_str == "gpt-4omini" or model_str == "gpt4o-mini":
                model_str = "gpt-4o-mini"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages, temperature=temp
                        )
                else:
                    client = OpenAI()
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18", messages=messages, )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content
            elif model_str == "claude-3.5-sonnet":
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-3-5-sonnet-latest",
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}])
                answer = json.loads(message.to_json())["content"][0]["text"]
            elif model_str == "gpt4o" or model_str == "gpt-4o":
                model_str = "gpt-4o"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages, temperature=temp)
                else:
                    client = OpenAI()
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-4o-2024-08-06", messages=messages, )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4o-2024-08-06", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content
            elif model_str == "deepseek-chat" or model_str == "deepseek-reasoner":
                # Normalize model name
                if model_str == "deepseek-chat":
                    deepseek_model = "deepseek-chat"
                elif model_str == "deepseek-reasoner":
                    deepseek_model = "deepseek-coder-instruct"
                
                # Format messages with system prompt
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
                
                if version == "0.28":
                    raise Exception("Please upgrade your OpenAI version to use DeepSeek client")
                else:
                    try:
                        # Initialize DeepSeek client
                        deepseek_client = OpenAI(
                            api_key=os.getenv('DEEPSEEK_API_KEY'),
                            base_url="https://api.deepseek.com/v1"
                        )
                        
                        # Create completion with appropriate parameters
                        completion_params = {
                            "model": deepseek_model,
                            "messages": messages,
                        }
                        
                        # Add temperature if specified
                        if temp is not None:
                            completion_params["temperature"] = temp
                            
                        # Make API call
                        completion = deepseek_client.chat.completions.create(**completion_params)
                        
                        # Extract answer
                        answer = completion.choices[0].message.content
                        
                        # Count tokens for cost tracking
                        if model_str not in TOKENS_IN:
                            TOKENS_IN[model_str] = 0
                            TOKENS_OUT[model_str] = 0
                            
                        # DeepSeek uses cl100k_base tokenizer (same as GPT-4)
                        encoding = tiktoken.get_encoding("cl100k_base")
                        
                        # Track token usage from API response if available
                        if hasattr(completion, 'usage') and completion.usage is not None:
                            TOKENS_IN[model_str] += completion.usage.prompt_tokens
                            TOKENS_OUT[model_str] += completion.usage.completion_tokens
                        else:
                            # Fallback to estimation
                            TOKENS_IN[model_str] += len(encoding.encode(system_prompt + prompt))
                            TOKENS_OUT[model_str] += len(encoding.encode(answer))
                            
                        return answer
                        
                    except Exception as e:
                        print(f"DeepSeek API error: {e}")
                        time.sleep(timeout)
                        continue
            elif model_str == "o1-mini":
                model_str = "o1-mini"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-mini-2024-09-12", messages=messages)
                answer = completion.choices[0].message.content
            elif model_str == "o1":
                model_str = "o1"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model="o1-2024-12-17",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-2024-12-17", messages=messages)
                answer = completion.choices[0].message.content
            elif model_str == "o1-preview":
                model_str = "o1-preview"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-preview", messages=messages)
                answer = completion.choices[0].message.content

            try:
                if model_str in ["o1-preview", "o1-mini", "claude-3.5-sonnet", "o1"]:
                    encoding = tiktoken.encoding_for_model("gpt-4o")
                elif model_str in ["deepseek-chat"]:
                    encoding = tiktoken.get_encoding("cl100k_base")
                else:
                    encoding = tiktoken.encoding_for_model(model_str)
                if model_str not in TOKENS_IN:
                    TOKENS_IN[model_str] = 0
                    TOKENS_OUT[model_str] = 0
                TOKENS_IN[model_str] += len(encoding.encode(system_prompt + prompt))
                TOKENS_OUT[model_str] += len(encoding.encode(answer))
                if print_cost:
                    print(f"Current experiment cost = ${curr_cost_est()}, ** Approximate values, may not reflect true cost")
            except Exception as e:
                if print_cost:
                    print(f"Cost approximation has an error? {e}")
            return answer
        except Exception as e:
            print("Inference Exception:", e)
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")


# print(query_model(model_str="o1-mini", prompt="hi", system_prompt="hey"))