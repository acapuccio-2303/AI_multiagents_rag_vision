from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.schema.runnable import RunnableLambda
from langchain.chains.combine_documents import create_stuff_documents_chain

def build_rag_chain(llm, vectorstore):
    """
    Costruisce la catena RAG con un vectorstore come retriever.
    """
    #PROMPT per riformulare la domanda usando la cronologia. 
    retriever_prompt = ChatPromptTemplate.from_messages([
        ("system", "Sei un assistente tecnico. Se utile, riformula la domanda usando la cronologia."),
        ("human", "{input}")
    ])
    
    #CREO IL RETRIEVER con cronologia conversazionale
    retriever = create_history_aware_retriever(
        llm=llm,
         retriever=vectorstore.as_retriever(
        search_type="mmr",          
        search_kwargs={"k": 6}),    
        prompt=retriever_prompt
    )

    #PROMPT finale per generare la risposta basata sul contesto (Per generare output umano leggendo contesto + domanda + storia)

    final_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Sei un assistente tecnico. Rispondi in modo mirato SOLO alla domanda usando il contesto fornito.\n"
         "- Se l'informazione NON è nel contesto, rispondi: 'Non lo so'.\n"
         "- Cita le fonti rilevanti con il loro 'source' (es. nome file) in fondo alla risposta.\n"),
        ("human", "Domanda: {input}\n\nContesto:\n{context}")
    ])


 # 👇 Template per ogni documento (usa sia testo che source) 
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="[Fonte: {source}]\n{page_content}"
    )

    # Catena di combinazione documenti
    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=final_prompt,
        document_variable_name="context",
        document_separator="\n---\n",
        document_prompt=document_prompt   
    )

    base_rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain
    )

    return base_rag_chain



