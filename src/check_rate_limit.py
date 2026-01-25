"""
Check OpenAI API rate limits for the current account.
"""
import os
from dotenv import load_dotenv
import httpx

load_dotenv()

def check_rate_limits():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    # Make a simple API call to get rate limit headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Use a minimal request to check limits
    data = {
        "model": "gpt-4.1",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1
    }

    print("Checking rate limits for GPT-4.1...")
    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=30
    )

    print("\n" + "=" * 50)
    print("Rate Limit Headers (GPT-4.1)")
    print("=" * 50)

    for key, value in response.headers.items():
        if "ratelimit" in key.lower() or "limit" in key.lower():
            print(f"{key}: {value}")

    # Check GPT-5.2
    data["model"] = "gpt-5.2"
    data["max_completion_tokens"] = 1
    del data["max_tokens"]

    print("\n" + "=" * 50)
    print("Checking rate limits for GPT-5.2...")
    print("=" * 50)

    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=30
    )

    for key, value in response.headers.items():
        if "ratelimit" in key.lower() or "limit" in key.lower():
            print(f"{key}: {value}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    check_rate_limits()
