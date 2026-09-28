import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

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


def generate_response(user_message: str) -> str:
    response = client.chat.completions.create(
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
    )

    generated_text = response.choices[0].message.content

    if not generated_text:
        raise RuntimeError("The LLM returned an empty response.")

    return generated_text