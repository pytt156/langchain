from langchain.agents import create_agent

from util.models import get_model
from util.streaming_utils import STREAM_MODES, handle_stream
from util.pretty_print import get_user_input
from util.tools import (
    list_files,
    find_files,
    read_file,
)

SYSTEM_PROMPT = """
<role>
You are a code review agent. You navigate and read a local codebase using tools.
</role>

<rules>
- NEVER guess or construct file paths manually.
- ALWAYS call find_files before read_file when given a filename.
- NEVER call read_file without a path returned from find_files or list_files.
- NEVER print tool calls as text. Use the actual tools.
- NEVER modify any files.
</rules>

<workflow>
1. User mentions a file, then call find_files with the filename
2. Use the path from find_files result, then call read_file with that exact path
3. Analyze and provide a review
</workflow>

<output>
**Assessment**: one-line summary
**Issues**: list of concrete problems
**Suggested Fix**: specific code changes
**Next Steps**: what to do next
</output>
""".strip()


def build_agent():
    model = get_model(temperature=0.1, top_p=0.8)

    return create_agent(
        model=model,
        tools=[
            list_files,
            find_files,
            read_file,
        ],
        system_prompt=SYSTEM_PROMPT,
        name="CodeAgent",
    )


def run():
    agent = build_agent()

    print("Code Review Agent")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = get_user_input(
            "What code did you fuck up this time?", agent_name="CodeAgent"
        )
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        chunks = agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode=STREAM_MODES,
        )
        handle_stream(chunks, agent_name="CodeAgent")


if __name__ == "__main__":
    run()
