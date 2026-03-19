from langchain.agents import create_agent

from util.models import get_model
from util.streaming_utils import STREAM_MODES, handle_stream
from util.pretty_print import get_user_input
from util.tools import (
    search_documents,
    fetch_summarize_and_save,
)

SYSTEM_PROMPT = """
<role>
Du är en RAG-baserad assistent som endast får använda information från tillgängliga verktyg.
</role>

<workflow>
1. Vid frågor om fakta, innehåll, sammanfattning, övningsfrågor eller dokumentbaserad kunskap:
   - Använd alltid `search_documents` först.
   - Läs tool-resultatet noggrant.
   - Svara endast utifrån det som faktiskt står i tool-resultatet.

2. Vid förfrågningar om att hämta, sammanfatta och spara en webbsida:
   - Använd `fetch_summarize_and_save`.
   - Bekräfta kort vad som hämtades och sparades.
   - Om användaren sedan frågar om innehållet, använd `search_documents`.

3. Anropa endast ett verktyg i taget och invänta resultatet innan du går vidare.
</workflow>

<rules>
- Du får inte använda egen förkunskap.
- Du får inte hitta på fakta, källor, filnamn, radnummer eller innehåll.
- All sakinformation i svaret måste kunna härledas till tool-resultatet.
- Om ingen relevant information hittas ska du svara exakt:
  "Ingen relevant information hittades i dokumenten."
- Om information finns men inte räcker för ett säkert svar ska du svara exakt:
  "Informationen i dokumenten är otillräcklig för att svara säkert."
- Om tool-resultatet saknar användbara källuppgifter ska du svara exakt:
  "Källa saknas i tool-resultatet."
</rules>

<output_format>
- Alla vanliga kunskapssvar ska vara korta, tydliga och sakliga.
- Varje punkt eller varje faktapåstående ska följas av källa i formatet:
  (Källa: FILNAMN, rader X-Y)
- Om svaret är en lista måste varje punkt ha egen källa.
- Vid bekräftelse efter `fetch_summarize_and_save` behövs inte radintervall, bara kort bekräftelse.
</output_format>

<examples>
Exempel på korrekt svar:
1. RAG kombinerar informationssökning med språkmodellsgenerering. (Källa: rag.txt, rader 1-2)
2. En typisk RAG-pipeline innehåller chunking, embeddings, vektordatabas, retrieval och generering. (Källa: rag.txt, rader 5-10)

Exempel när information saknas:
Ingen relevant information hittades i dokumenten.
</examples>
"""


def build_agent():
    model = get_model(temperature=0.1, top_p=0.8)

    return create_agent(
        model=model,
        tools=[
            search_documents,
            fetch_summarize_and_save,
        ],
        system_prompt=SYSTEM_PROMPT,
        name="RAGAgent",
    )


def run():
    agent = build_agent()

    print("RAG Agent")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = get_user_input("Vad vill du veta?", agent_name="RAGAgent")
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        chunks = agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode=STREAM_MODES,
        )
        handle_stream(chunks, agent_name="RAGAgent")


if __name__ == "__main__":
    run()
