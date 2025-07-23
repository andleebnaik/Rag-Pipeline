import json
import logging
import traceback
import os
from openai import OpenAI
from dotenv import load_dotenv

class LLMService:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('TEAMIFIED_OPENAI_API_KEY')
        self.model = os.getenv('LLM_MODEL')
        self.client = OpenAI(api_key=self.api_key)
    
    def completion(self, user_prompt, system_prompt):
        try:
            response = self.client.chat.completions.create(
                store=False,
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error while LLM call: {traceback.format_exc()}")
            return None

    def generate_response(self, query_object=None):
        try:
             # Get the absolute path of prompt_studio.json
            current_dir = os.path.dirname(__file__)  # This gets the directory of response_generation.py
            prompt_path = os.path.join(current_dir, "..", "prompt_studio.json")
            prompt_path = os.path.abspath(prompt_path)
            logging.info(f"Prompt file path: {prompt_path}")
            with open(prompt_path) as f:
                prompts = json.load(f)
                
            system_prompt = prompts["system_prompt"]
            user_prompt = prompts["user_prompt"] + f"\n\nUser Query: {query_object['query']}\n\nReferences : {query_object['references']}" if query_object else user_prompt
            
            response = self.completion(user_prompt, system_prompt)
            logging.info(f"RESPONSE QUERY: {response}")
            return response
        except Exception as e:
            logging.error(f"Error generating response: {traceback.format_exc()}")
            return None
