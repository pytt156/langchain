  # README
Title
=====

Langchain Demo Project


Overview
========

This project demonstrates the capabilities of Langchain, a framework for building conversational AI applications. The project includes several agents that can be used to perform various tasks such as code review, knowledge base maintenance, and documentation generation.


Agents
======

The project includes three agents:

*   `CodeAgent`: A code review agent that can navigate and read a local codebase using tools.
*   `RAGAgent`: A knowledge base maintenance agent that can search, add documents to, and answer questions from a local knowledge base.
*   `ReadmeAgent`: A documentation agent that gathers information by calling tools and produces a README.


Tech Stack
==========

The project uses the following technologies:

*   Langchain: A framework for building conversational AI applications.
*   Ollama: A chat model used for generating human-like text.
*   Python: The programming language used for developing the agents.

Project Structure
=================

The project is structured as follows:

*   `my_agents/`: Directory containing the agent code.
    *   `code_agent.py`: Code review agent.
    *   `rag_maintenance.py`: Knowledge base maintenance agent.
    *   `readme_agent.py`: Documentation agent.
*   `util/`: Directory containing utility functions.
    *   `models.py`: Functions for getting a chat model using Ollama.

Getting Started
===============

To get started with the project, follow these steps:

1.  Install the required dependencies by running `uv sync`.
2.  Set up the environment variables by creating a `.env` file and adding the necessary values.
3.  Run the agents `python -m my_agents.code_agent` e.g.

Usage
=====

To use the agents, follow these steps:

1.  CodeAgent: Provide the name of the file to review as input.
2.  RAGAgent: Provide a URL or paste the content to add to the knowledge base.
3.  ReadmeAgent: Run the agent and provide input to generate the README.

---

NOTES
====
This readme was created by the readme_agent(except for the next part):

## Human reflection

Valet av agenter utgick från vad som kändes mest användbart för mig just nu. En kodgranskare för att slippa läsa igenom filer manuellt, en kunskapsbasagent för att kunna spara och söka i egna dokument, och en README-generator eftersom det är något man annars tenderar att skjuta upp.

Det svåraste var att få readme-agenten att faktiskt köra klart alla steg innan den började skriva. Llama vill gärna "tänka högt" mellan verktygsanropen, vilket ledde till en hel del prompt-felsökning. Bland annat tolkade den OUTPUT: som ett verktygsnamn och försökte anropa det, något jag inte tänkte på innan.

I efterhand hade en enkel pipeline utan agent förmodligen fungerat bättre för just readme-agenten, eftersom stegen alltid är desamma. Agenter är mer värda när flödet faktiskt behöver anpassa sig efter vad som händer.

Roligaste agenten att bygga var RAG-agenten, det kändes mest "på riktigt" att kunna mata in URL:er och sedan ställa frågor om innehållet.