import openai
import os

class PromptFunction:
    def __init__(self, prompt, max_tokens=2, model="gpt-4o-mini"):
        self.prompt = prompt.strip()
        self.max_tokens = max_tokens
        self.model = model

        # Optionally set your key from environment variable
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def __call__(self, input_text):
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": input_text.strip()}
        ]
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0.0,
            stop=["\n"]
        )
        return response["choices"][0]["message"]["content"].strip()
