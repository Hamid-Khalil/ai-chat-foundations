"""Manual "first contact" check — run from the project root:

    python -m scripts.smoke_llm

Uses the synchronous `invoke` because a plain script has no event loop. This is the
ONLY place `invoke` is allowed; inside `app/` always use `await ainvoke`.
"""

from langchain.messages import HumanMessage

from app.core.config import get_settings
from app.core.llm import get_chat_model


def main() -> None:
    settings = get_settings()
    settings.require_api_key()

    # Task 0.2 proof: the key must print as '**********', never the real value.
    print("settings:", settings)

    # Task 1.1 proof: repeated calls return the very same cached object.
    print("cache works (a is b):", get_chat_model() is get_chat_model())

    model = get_chat_model()
    response = model.invoke([HumanMessage("Say hello in one short sentence.")])

    print("\n--- response.content ---")
    print(type(response.content), response.content)
    print("\n--- response.text ---")
    print(type(response.text), str(response.text))
    print("\n--- response.usage_metadata ---")
    print(response.usage_metadata)
    print("\n--- response.response_metadata ---")
    print(response.response_metadata)


if __name__ == "__main__":
    main()
