import asyncio
import warnings

from adk_agents.agent import root_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

warnings.filterwarnings("ignore", category=UserWarning)


APP_NAME = "clinical_ai"
USER_ID = "manoj"
SESSION_ID = "test_session"


async def main():
    session_service = InMemorySessionService()

    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    runner = Runner(
        app_name=APP_NAME,
        agent=root_agent,
        session_service=session_service,
    )

    question = (
        "Top 3 care gaps for patients who have both Diabetes and Hypertension "
    )

    message = Content(
        role="user",
        parts=[Part(text=question)],
    )

    print(f"\nQuestion: {question}\n")
    print("Running ADK agent...\n")

    final_text = []

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=message,
    ):
        # Helpful debug line. Keep it while testing.
        print(f"[event] author={getattr(event, 'author', None)}")

        if not event.content or not event.content.parts:
            continue

        for part in event.content.parts:
            text = getattr(part, "text", None)
            if text:
                print(text)
                final_text.append(text)

            function_call = getattr(part, "function_call", None)
            if function_call:
                print(f"[tool call] {function_call.name}: {function_call.args}")

            function_response = getattr(part, "function_response", None)
            if function_response:
                print(f"[tool response] {function_response.name}")

    print("\nDone.\n")

    if final_text:
        print("Final collected text:")
        print("\n".join(final_text))
    else:
        print("No text response was produced. Check tool calls/errors above.")


if __name__ == "__main__":
    asyncio.run(main()) 