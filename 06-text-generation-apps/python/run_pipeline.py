import asyncio
import warnings
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from adk_agents.agent import root_agent

load_dotenv()
warnings.filterwarnings("ignore")

APP_NAME = "clinical_ai"
USER_ID = "manoj"
SESSION_ID = "ingestion_session_01"


async def main():
    # Use the same pattern as your working run_adk_agent.py
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

    print("====================================================")
    print("🏥 Clinical AI Orchestration & Ingestion Pipeline")
    print("====================================================")
    print("Examples:")
    print("  - Ingest the file: data/synthea/patients.csv")
    print("  - Find care gaps for patients with Diabetes")
    print("  - Search clinical notes for insomnia")
    print("Type 'exit' or 'quit' to end session.")
    print("====================================================\n")

    while True:
        try:
            prompt = input("Orchestrator > ").strip()

            if not prompt:
                continue

            if prompt.lower() in ["exit", "quit"]:
                print("Ending session. Goodbye!")
                break

            message = Content(
                role="user",
                parts=[Part(text=prompt)],
            )

            print(f"\n[Sending to Orchestrator...]\n")

            async for event in runner.run_async(
                user_id=USER_ID,
                session_id=SESSION_ID,
                new_message=message,
            ):
                if not event.content or not event.content.parts:
                    continue

                for part in event.content.parts:
                    text = getattr(part, "text", None)
                    if text:
                        print(text)

                    function_call = getattr(part, "function_call", None)
                    if function_call:
                        print(f"  → Tool Call: {function_call.name}: {function_call.args}")

                    function_response = getattr(part, "function_response", None)
                    if function_response:
                        print(f"  ← Tool Result: {function_response.name}")

            print()

        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n[Runner Error]: {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())