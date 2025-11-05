from collections.abc import Iterable
from botocore.config import Config
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from typing import List, Union, Tuple, Any
from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
from google.genai import Client
from google.genai import types
from mistralai import Mistral
import os
import json
import time
from openai import OpenAI
from typing import Union, List
from botocore.exceptions import ClientError
import boto3
from dotenv import load_dotenv
load_dotenv(override=True)


def get_env_variable(var_name):
    """Fetch environment variable or return None if not set."""
    return os.getenv(var_name)


CALL_SLEEP = 60
clients = {}


def initialize_clients():
    """Dynamically initialize available clients based on environment variables."""
    try:
        gpt_api_key = get_env_variable('OPENAI_API_KEY')
        if gpt_api_key:
            clients['gpt'] = OpenAI(api_key=gpt_api_key)

        claude_api_key = get_env_variable('CLAUDE_API_KEY')
        claude_base_url = get_env_variable('BASE_URL_CLAUDE')
        if claude_api_key and claude_base_url:
            clients['claude'] = OpenAI(
                base_url=claude_base_url, api_key=claude_api_key)

        deepseek_api_key = get_env_variable('DEEPSEEK_API_KEY')
        deepseek_base_url = get_env_variable('BASE_URL_DEEPSEEK')
        if deepseek_api_key and deepseek_base_url:
            clients['deepseek'] = OpenAI(
                base_url=deepseek_base_url, api_key=deepseek_api_key)

        deepinfra_api_key = get_env_variable('DEEPINFRA_API_KEY')
        deepinfra_base_url = get_env_variable('BASE_URL_DEEPINFRA')
        if deepinfra_api_key and deepinfra_base_url:
            clients['deepinfra'] = OpenAI(
                base_url=deepinfra_base_url, api_key=deepinfra_api_key)

        gemini_api_key = get_env_variable('GEMINI_API_KEY')
        # gemini_base_url = get_env_variable('BASE_URL_GEMINI')
        if gemini_api_key:
            clients['gemini'] = Client(api_key=gemini_api_key)
        mistral_api_key = get_env_variable('MISTRAL_API_KEY')
        if mistral_api_key:
            clients['mistral'] = Mistral(api_key=mistral_api_key)
        aws_access_key_id = get_env_variable('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = get_env_variable('AWS_SECRET_ACCESS_KEY')
        aws_region_name = get_env_variable('AWS_REGION_NAME')
        if aws_secret_access_key and aws_access_key_id and aws_region_name:
            config = Config(read_timeout=3600)
            clients['bedrock-east-1'] = boto3.client(
                "bedrock-runtime", region_name='us-east-1',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                config=config
            )
            clients['bedrock-west-2'] = boto3.client(
                "bedrock-runtime", region_name='us-west-2',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                config=config
            )
            # if not clients:
        #     print("No valid API credentials found. Exiting.")
        #     exit(1)
    except Exception as e:
        print(f"Error during client initialization: {e}")
        exit(1)


def get_client(model_name):
    """Select appropriate client based on the given model name."""
    initialize_clients()
    if 'gpt-5' in model_name or 'o1-' in model_name:
        client = clients.get('gpt')
    # elif 'claude' in model_name:
    #     client = clients.get('claude')
    # elif 'deepseek' in model_name:
    #     client = clients.get('deepseek')
    # elif any(keyword in model_name.lower() for keyword in ['llama', 'nova-premier', 'microsoft']):
    #     client = clients.get('deepinfra')
    elif 'gemini' in model_name.lower():
        client = clients.get('gemini')
    # elif 'mistral' in model_name.lower() or 'mixtral' in model_name.lower():
    #     client = clients.get('mistral')
    elif any(name in model_name.lower() for name in ['nova-premier', 'mistral', 'llama', 'deepseek', 'claude', 'gpt-oss']):
        if 'gpt-oss' in model_name.lower() or 'mistral' in model_name.lower():
            client = clients.get('bedrock-west-2')
        else:
            client = clients.get('bedrock-east-1')
    else:
        raise ValueError(f"Unsupported or unknown model name: {model_name}")

    if not client:
        raise ValueError(f"{model_name} client is not available.")
    return client


def get_model_tokenizer(model_name, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=cache_dir, device_map="auto", torch_dtype="auto")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def read_prompt_from_file(filename):
    with open(filename, 'r') as file:
        prompt = file.read()
    return prompt


def read_data_from_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


def parse_json(output):
    try:
        output = ''.join(output.splitlines())
        if '{' in output and '}' in output:
            start = output.index('{')
            end = output.rindex('}')
            output = output[start:end + 1]
        data = json.loads(output)
        return data
    except Exception as e:
        print("parse_json:", e)
        return None


def check_file(file_path):
    if os.path.exists(file_path):
        return file_path
    else:
        raise IOError(f"File not found error: {file_path}.")


def update_to_bedrock_format(messages):
    contents = []
    for message in messages:
        text = message["content"]
        role = message["role"]
        contents.append({
            'role': role,
            'content': [{'text': text}]
        })
    return contents


def gpt_call(client, query: Union[List, str], model_name="gpt-4o", temperature=0, max_tokens=4096, json_format=False):
    if isinstance(query, List):
        messages = query
    elif isinstance(query, str):
        messages = [{"role": "user", "content": query}]
    if "gemini" in model_name.lower():
        contents = update_to_gemini_format(
            messages)
        config = types.GenerateContentConfig(
            candidate_count=1,
            max_output_tokens=max_tokens,
            temperature=temperature,
            response_mime_type='text/plain' if not json_format else 'application/json',
            safety_settings=[
                types.SafetySetting(
                    category='HARM_CATEGORY_HATE_SPEECH',
                    threshold='BLOCK_NONE'
                ),
                types.SafetySetting(
                    category='HARM_CATEGORY_HARASSMENT',
                    threshold='BLOCK_NONE'
                ),
                types.SafetySetting(
                    category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                    threshold='BLOCK_NONE'
                ),
                types.SafetySetting(
                    category='HARM_CATEGORY_DANGEROUS_CONTENT',
                    threshold='BLOCK_NONE'
                ),
            ]
        )
    elif any(name in model_name.lower() for name in ['nova-premier', 'mistral', 'llama', 'deepseek', 'claude', 'gpt-oss']):
        messages = update_to_bedrock_format(
            messages)
    for _ in range(5):
        full_info = ""
        try:
            if "gemini" in model_name.lower():
                response = client.models.generate_content(
                    model=model_name, contents=contents, config=config)
                full_info = to_serializable(response)
                resp = response.text
            elif any(name in model_name.lower() for name in ['nova-premier', 'mistral', 'llama', 'deepseek', 'claude', 'gpt-oss']):
                model_response = client.converse(
                    modelId=model_name,
                    messages=messages,
                    inferenceConfig={
                        'temperature': temperature,
                        'maxTokens': max_tokens,
                    }
                )
                full_info = model_response
                if 'gpt-oss' in model_name.lower():
                    resp = model_response['output']['message']['content'][1]['text']
                else:
                    resp = model_response['output']['message']['content'][0]['text'] or ""
            elif "gpt-5" in model_name.lower():
                max_tokens = 8192
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    # temperature=temperature,
                    max_completion_tokens=max_tokens

                )
                full_info = completion

                print(completion)
                resp = completion.choices[0].message.content
                # elif 'mistral' in model_name.lower() or 'mixtral' in model_name.lower():
                #     completion = client.chat.complete(
                #         model=model_name,
                #         messages=messages,
                #         temperature=temperature,
                #         max_tokens=max_tokens,
                #         response_format={
                #             "type": "json_object" if json_format else "text",
                #         }
                #     )
                #     resp = completion.choices[0].message.content
                #     print(resp)
            elif 'o1-' in model_name:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens
                )
                resp = completion.choices[0].message.content
            else:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    response_format={
                        "type": "json_object" if json_format else "text",
                    }
                )
                resp = completion.choices[0].message.content
            # if resp == "":
            #     time.sleep(CALL_SLEEP)
            #     continue
            return resp, full_info
        except Exception as e:
            print(f"GPT_CALL Error: {model_name}:{e}")
            print("full_info", full_info)
            time.sleep(CALL_SLEEP)
            continue
    return "", full_info


def gpt_call_append(client, model_name, dialog_hist: List, query: str, temperature: float = 0, json_format=False):
    dialog_hist.append({"role": "user", "content": query})
    resp, full_info = gpt_call(client, dialog_hist, model_name=model_name,
                               temperature=temperature, json_format=json_format)
    dialog_hist.append(
        {"role": "assistant", "content": resp})
    return resp, full_info, dialog_hist


def collect_queries(data):
    q = [query
         for actor in data.get('actors', [])
         for query in actor.get('queries', [])]
    return q


def get_runner(model_name, is_api=True, cache_dir=None):
    if is_api:
        runner = get_client(model_name=model_name)
    else:
        runner = get_model_tokenizer(
            model_name=model_name, cache_dir=cache_dir)
    return runner


def is_reject(query):
    reject_mark = ["i'm sorry", "i cannot", "i can't",
                   "the generated text has been blocked by our content filters"]
    if any(mark.lower() in query.lower() for mark in reject_mark):
        return True
    else:
        return False
