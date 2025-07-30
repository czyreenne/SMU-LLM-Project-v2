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
        "gpt-4.1-mini": "openai",
        "gpt-4.1": "openai",
        "o3": "openai",
        "o4-mini": "openai",
        "claude-3-5-sonnet": "anthropic",
        "claude-sonnet-4": "anthropic",
        "claude-opus-4": "anthropic",
        "deepseek-chat": "deepseek",
    }
    return provider_map.get(model, "unknown")

def curr_cost_est():
    million_tokens = 1000000
    costmap_in = {
        "gpt-4o": 2.50 / million_tokens,
        "gpt-4o-mini": 0.150 / million_tokens,
        "gpt-4.1-mini": 0.40 / million_tokens,
        "gpt-4.1": 2.00 / million_tokens,
        "o3": 2.00 / million_tokens,
        "o4-mini": 1.10 / million_tokens,
        "claude-3-5-sonnet": 3.00 / million_tokens,
        "claude-sonnet-4": 3.00 / million_tokens,
        "claude-opus-4": 15.00 / million_tokens,
        "deepseek-chat": 1.00 / million_tokens,
    }
    costmap_out = {
        "gpt-4o": 10.00/ million_tokens,
        "gpt-4o-mini": 0.6 / million_tokens,
        "gpt-4.1-mini": 1.60 / million_tokens,
        "gpt-4.1": 8.00 / million_tokens,
        "o3": 8.00 / million_tokens,
        "o4-mini": 4.40 / million_tokens,
        "claude-3-5-sonnet": 15.00 / million_tokens,
        "claude-sonnet-4": 15.00 / million_tokens,
        "claude-opus-4": 75.00 / million_tokens,
        "deepseek-chat": 5.00 / million_tokens,
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
            elif model_str == "gpt-4.1-mini" or model_str == "gpt4.1-mini":
                model_str = "gpt-4.1-mini"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",
                            messages=messages, temperature=temp)
                else:
                    client = OpenAI()
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-4.1-mini-2025-04-14", messages=messages)
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4.1-mini-2025-04-14", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content
            elif model_str == "gpt-4.1" or model_str == "gpt4.1":
                model_str = "gpt-4.1"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",
                            messages=messages, temperature=temp)
                else:
                    client = OpenAI()
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-4.1-2025-04-14", messages=messages)
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4.1-2025-04-14", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content
            elif model_str == "claude-3.5-sonnet":
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-3-5-sonnet-latest",
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}])
                answer = json.loads(message.to_json())["content"][0]["text"]
            elif model_str == "claude-sonnet-4":
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}])
                answer = json.loads(message.to_json())["content"][0]["text"]
            elif model_str == "claude-opus-4":
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-opus-4-20250514",
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}])
                answer = json.loads(message.to_json())["content"][0]["text"]
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
                if model_str in ["o1-preview", "o1-mini", "claude-3.5-sonnet", "o1", 
                                 "gpt-4.1", "gpt-4.1-mini", "o3", "o4-mini"]:
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