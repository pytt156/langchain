from langchain.agents import create_agent

from util.models import get_model, AvailableModels
from util.pretty_print import get_user_input
from util.streaming_utils import STREAM_MODES, handle_stream

from util.tools import (
    calculate,
    fetch_summarize_and_save,
    get_current_time,
    get_web_search_tool,
    read_file,
    write_file,
    replace_in_file,
    search_documents,
    list_files,
    index_project,
    search_codebase,
)


SYSTEM_PROMPT = """
<context>
You are a precise coding assistant operating inside a local project.

Project root:
 /home/pytt/skola/langchain

When listing files or indexing the project, use that path unless the user explicitly provides another.
You have access to tools that can read files, search the codebase,
modify files, and retrieve documents.
</context>

<rules>
Always prefer tools over guessing.

When working with code:
1. If you do not know the file location, use list_files.
2. If the codebase is not indexed, run index_project.
3. Use search_codebase to locate relevant code.
4. Read the file before suggesting modifications.
5. Only modify files when explicitly asked.

Never invent file contents.
When information comes from tools, trust the tool output.
Be concise.
</rules>

<tone>
Dry, slightly sarcastic, but helpful.
</tone>

<output>
Short and factual answers.
</output>
""".strip()


def build_agent():
    """Create and configure the LangChain agent."""
    model = get_model(
        model_name=AvailableModels.LLAMA_70B,
        temperature=0.2,
        top_p=0.9,
    )

    web_tools = get_web_search_tool()

    tools = [
        calculate,
        fetch_summarize_and_save,
        get_current_time,
        read_file,
        write_file,
        replace_in_file,
        search_documents,
        list_files,
        index_project,
        search_codebase,
        *web_tools,
    ]

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent


def run():
    """Interactive CLI loop."""
    agent = build_agent()
    messages = []

    print("\nLangChain code assistant ready.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = get_user_input("What do you want?")

        if not user_input:
            continue

        if user_input.strip().lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        try:
            process_stream = agent.stream(
                {"messages": messages},
                stream_mode=STREAM_MODES,
            )

            final_response = handle_stream(process_stream)

            if final_response:
                messages.append(
                    {
                        "role": "assistant",
                        "content": final_response,
                    }
                )

        except KeyboardInterrupt:
            print("\nInterrupted.")
            break

        except Exception as exc:
            error_message = f"Agent error: {exc}"
            print(error_message)

            messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )


if __name__ == "__main__":
    run()
