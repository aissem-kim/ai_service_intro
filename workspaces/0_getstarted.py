import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

# response = client.responses.create(
#     model="gpt-5.6",
#     input="Write a one-sentence bedtime story about a unicorn.",
# )
response = client.responses.create(
    model="gpt-4.1-mini",
    input="AI에 대해 간단하게 설명해줘.",
)

print(response.output_text)