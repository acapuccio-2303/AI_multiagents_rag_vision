from typing import TypedDict
from ..agents.agent_state import AgentState
from langgraph.graph import StateGraph, END
from ..agents.agents import router, agente_switch, run_tecnico, run_generale, run_vision
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# === ORCHESTRAZIONE E ROUTING ===


def build_graph(llm=None, vision_model=None, logger=None) -> StateGraph:
    """Crea e restituisce il grafo dello stato con gli agenti"""

    builder = StateGraph(AgentState)
    
    #Aggiungo i nodi al grafo + router come punto di ingresso
    builder.add_node("router", lambda state: router(state, llm, logger=logger))
    builder.add_node("tecnico", lambda state: run_tecnico(state, llm, logger=logger))
    builder.add_node("generale", lambda state: run_generale(state, llm, logger=logger))
    builder.add_node("vision", lambda state: run_vision(state, vision_model, logger=logger))
    builder.set_entry_point("router")

    # Condizioni per uscire dal router verso agente tecnico o generale
    builder.add_conditional_edges("router", agente_switch, {
        "tecnico": "tecnico",
        "generale": "generale",
        "vision": "vision"
    })

    # Entrambi gli agenti portano all'END (fine flusso)
    builder.add_edge("tecnico", END)
    builder.add_edge("generale", END)
    builder.add_edge("vision", END)

    # Compilazione definitiva del grafo
    grafo = builder.compile()
    logger.info("✅ Grafo degli agenti creato con successo!")
    logger.info("--------")
    return grafo


