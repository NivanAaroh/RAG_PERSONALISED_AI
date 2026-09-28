import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("GROQ_API_KEY")
model = os.getenv("GROQ_MODEL")

if not api_key:
    raise RuntimeError("GROQ_API_KEY is not configured.")

if not model:
    raise RuntimeError("GROQ_MODEL is not configured.")


client = Groq(api_key=api_key)


SYSTEM_PROMPT = (
    "You are a helpful AI assistant.\n"
    "Answer the user's question clearly and concisely."
)


def generate_response(user_message: str):
    response_stream = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        stream=True,
    )

    for chunk in response_stream:
        if not chunk.choices:
            continue

        content = chunk.choices[0].delta.content

        if content:
            yield content