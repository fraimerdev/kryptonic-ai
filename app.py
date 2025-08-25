import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_mongodb import MongoDBChatMessageHistory, MongoDBAtlasVectorSearch
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.tools import tool
from flask import Flask, render_template, request, jsonify
from db import embeddings as embeddings_model

load_dotenv()

app = Flask(__name__)

llm = ChatOpenAI(model="gpt-4o-mini")

embeddings = OpenAIEmbeddings()

vector_store = MongoDBAtlasVectorSearch(
    embedding=embeddings, collection=embeddings_model, index_name="vector_index"
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are "Kryptonic," a young, super intelligent cryptocurrency expert with a Gen Z persona. Your goal is to be an informative, friendly, encouraging, and fun guide for beginners learning about crypto. You should use Gen Z slang but still sound professional and knowledgeable with a casual, conversational tone. Your responses should be easy to understand and avoid overwhelming jargon.  

            Rules
            Do not answer any non-related cryptocurrency questions
            Never be rude or disrespectful
            Never use inappropriate language
            
            if asked, respond in a respectful manner, explaining that the user`s question is not a part of its function
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{query}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


@tool
def retrieve(query: str) -> str:
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    return retrieved_docs


agent = create_openai_tools_agent(
    llm=llm,
    tools=[retrieve],
    prompt=prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=[retrieve],
    verbose=True,
)

agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: MongoDBChatMessageHistory(
        session_id=session_id,
        connection_string=os.getenv("MONGODB_URL"),
        database_name="kryptonic",
        collection_name="history",
        session_id_key="session_id",
        history_key="history",
    ),
    input_messages_key="query",
    history_messages_key="history",
)


@app.route("/")
def index():
    """Renders the main page of the application."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """Handles chat messages from the front-end and returns a response."""
    session_id = request.json["session_id"] or "testing"
    response = agent_with_history.invoke(
        {"query": request.json["query"]},
        config={"configurable": {"session_id": session_id}},
    )
    return jsonify({"output": response["output"], "session_id": session_id})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
