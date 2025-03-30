import time
import os
import openai
from google import genai
#import google.generativeai as genai 
from configs import OPENAI_API_KEY
from transformers import *
import torch

class Call_OpenAI:
    def __init__(self, model="gpt-4o-mini"):
        self.count = 0
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
    def __repr__(self):
        return (f'Model: {self.model}, Api Calls: {self.count}, '
                f'Input Tokens: {self.total_input_tokens}, '
                f'Output Tokens: {self.total_output_tokens}, '
                f'Total Tokens: {self.total_tokens}')

    def call(self, prompt):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a data scientist expert in SQL and LLM prompts."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
            temperature=0.01
        )
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += total_tokens
        print(f"API Call #{self.count+1} - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens} tokens")

        self.count += 1
        return response.choices[0]['message']['content'].strip()

class Call_Gemini:
    def __init__(self, model_name='gemini-2.0-flash-exp'):
        self.gemini_api_keys = []
        self.model = model_name
        self.count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        # Models:       'gemini-1.5-pro-latest', 'gemini-2.0-flash-exp'
        
    def __repr__(self):
        return (f'Model: {self.model}, Api Calls: {self.count}, '
                f'Input Tokens: {self.total_input_tokens}, '
                f'Output Tokens: {self.total_output_tokens}, '
                f'Total Tokens: {self.total_tokens}')

    def call(self, prompt):
        n = len(self.gemini_api_keys)
        generation_config = {
            "temperature": 0.01,
            "max_output_tokens": 512,
            "response_mime_type": "text/plain",
        }
        api_key = self.gemini_api_keys[self.count % n]
        client = genai.Client(api_key=api_key)
        local = 0
        
        while True:
            try:
                # Count tokens before generating content
                token_request = client.count_tokens(
                    model=self.model,
                    contents=prompt
                )
                input_tokens = token_request.total_tokens
                self.total_input_tokens += input_tokens
                
                # Generate content
                response = client.models.generate_content(
                    model=self.model, contents=prompt, generation_config=generation_config
                )
                
                # Count output tokens (this is approximate since Gemini API doesn't directly provide this)
                output_text = response.text.strip()
                output_token_request = client.count_tokens(
                    model=self.model,
                    contents=output_text
                )
                output_tokens = output_token_request.total_tokens
                self.total_output_tokens += output_tokens
                
                # Calculate total tokens
                total_tokens = input_tokens + output_tokens
                self.total_tokens += total_tokens
                
                # Print token usage
                print(f"API Call #{self.count+1} - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens} tokens")
                
                self.count += 1
                break
            except Exception as e:
                local += 1
                print(f'API key exhausted: {api_key}')

                api_key = self.gemini_api_keys[local % n]
                client = genai.Client(api_key=api_key)

        return output_text


class Call_DeepSeek:
    def __init__(self, model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"):
        self.count = 0
        self.model = model
        # Models:       deepseek/deepseek-r1-distill-llama-70b
        openai.api_base = "https://api.deepinfra.com/v1/openai"
        openai.api_key = ""

    def __repr__(self):
        return f'Model: {self.model}, Api Calls: {self.count}'
    def call(self, prompt):
        config = {
            'model': self.model,
            'messages': [
                {"role": "system", "content": "You are a data scientist."},
                {"role": "user", "content": prompt}
            ],
            'max_tokens': 512,
            'temperature': 0.01  # Deterministic
        }
        try:
            response = openai.ChatCompletion.create(**config)
            self.count+=1
        except Exception as e:
            print(e)

        return response.choices[0].message.content.strip()