import asyncio

from groq import AsyncGroq


async def main() -> None:
    # Reads GROQ_API_KEY automatically from the environment.
    async with AsyncGroq(timeout=30.0, max_retries=0) as client:
        response = await client.chat.completions.create(
            model="openai/gpt-oss-20b",
            reasoning_effort="low",
            max_completion_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": "Reply with exactly: Connection successful.",
                }
            ],
        )

        text = response.choices[0].message.content
        if not text:
            raise RuntimeError("The model returned no final answer.")

        print(text)


if __name__ == "__main__":
    asyncio.run(main())
