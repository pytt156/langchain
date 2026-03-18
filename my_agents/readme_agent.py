from langchain.agents import create_agent

from util.models import get_model
from util.streaming_utils import STREAM_MODES, handle_stream
from util.pretty_print import get_user_input
from util.tools import (
    list_files,
    index_project,
    search_codebase,
    read_file,
)

SYSTEM_PROMPT = """
<role>
You are a documentation agent. You gather information by calling tools, then produce a README.
</role>

<rules>
- Call tools one at a time. Wait for each result before calling the next.
- Do NOT write any text between tool calls - no commentary, no observations, no summaries.
- Your only text output is the finished README, written after all tools are done.
- Never invent file names, module names, features, or commands.
- Base every README section strictly on tool results.
</rules>

<workflow>
Work through this checklist from top to bottom. Call the next tool as soon as the previous one returns. Do not stop until every box is checked.

[ ] 1. list_files              - discover all project files
[ ] 2. index_project           - build the search index
[ ] 3. search_codebase         - query: "project purpose features tech stack run command agents"
[ ] 4. read_file pyproject.toml
[ ] 5. read_file each .py file found inside my_agents/ (one call per file)
[ ] 6. read_file util/models.py
[ ] 7. When all boxes above are checked: write the complete README markdown as your final text response.
</workflow>

<writing_rules>
- Use real module paths for run commands (e.g. python -m my_agents.code_agent).
- The README must be professional, concise, and GitHub-friendly.
- Omit any section you found no supporting information for.
</writing_rules>

<required_sections>
- Title
- Overview
- Agents
- Tech Stack
- Project Structure
- Getting Started
- Usage
</required_sections>
""".strip()


def build_agent():
    model = get_model(temperature=0, top_p=0.9)
    print(f"Using model: {model.model}")

    agent = create_agent(
        model=model,
        tools=[
            list_files,
            index_project,
            search_codebase,
            read_file,
        ],
        system_prompt=SYSTEM_PROMPT,
        name="ReadmeAgent",
    )
    return agent.with_config({"recursion_limit": 100})


def run():
    agent = build_agent()

    print("README Agent")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = get_user_input("Read it and weep?", agent_name="ReadmeAgent")
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        chunks = agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode=STREAM_MODES,
            config={"recursion_limit": 100},
        )
        handle_stream(chunks, agent_name="ReadmeAgent")


if __name__ == "__main__":
    run()
