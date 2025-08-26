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

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Initialize OpenAI language model with GPT-4o-mini
llm = ChatOpenAI(model="gpt-4o-mini")

# Initialize OpenAI embeddings for vector search
embeddings = OpenAIEmbeddings()

# Set up MongoDB Atlas vector store for semantic search
vector_store = MongoDBAtlasVectorSearch(
    embedding=embeddings, collection=embeddings_model, index_name="vector_index"
)

# Define the system prompt that shapes Kryptonic AI's personality and behavior
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are "Kryptonic AI," a young, super intelligent cryptocurrency expert with a Gen Z persona. Your goal is to be an informative, friendly, encouraging, and fun guide for beginners learning about crypto. You should use Gen Z slang but still sound professional and knowledgeable with a casual, conversational tone. Your responses should be easy to understand and avoid overwhelming jargon.  

            Rules
            Do not answer any non-related cryptocurrency questions
            Never be rude or disrespectful
            Never use inappropriate language
            Use the appropraite tools to find and answer relevant to a query
            If the user responds in a certain language you respond in the same language or if they request for you to respond in a certain language, 
            you respond in that language. Do not continue responding in the previous language, just respond in whatever language
            they texted in
            Keep the responeses not too long but impactful and make sure the response is informative.
            
            
            if asked, respond in a respectful manner, explaining that the user`s question is not a part of its function
            """,
        ),
        MessagesPlaceholder(variable_name="history"),  # Placeholder for conversation history
        ("human", "{query}"),  # User's current query
        MessagesPlaceholder(variable_name="agent_scratchpad"),  # Agent's reasoning process
    ]
)


@tool
def retrieve(query: str) -> str:
    """Retrieve information related to a query."""
    # Perform similarity search in the vector store to find relevant crypto information
    retrieved_docs = vector_store.similarity_search(query, k=2)
    return retrieved_docs


# Create the OpenAI tools agent with the LLM, tools, and prompt
agent = create_openai_tools_agent(
    llm=llm,
    tools=[retrieve],
    prompt=prompt,
)

# Create agent executor to run the agent with tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=[retrieve],
    verbose=True,  # Enable verbose logging for debugging
)

# Wrap agent with conversation history management using MongoDB
agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    # Lambda function to create MongoDB chat history instance for each session
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
    # Get session ID from request, default to "testing" if not provided
    session_id = request.json["session_id"] or "testing"
    
    # Invoke the agent with the user's query and session configuration
    response = agent_with_history.invoke(
        {"query": request.json["query"]},
        config={"configurable": {"session_id": session_id}},
    )
    
    # Return the agent's response and session ID as JSON
    return jsonify({"output": response["output"], "session_id": session_id})


if __name__ == "__main__":
    # Run the Flask application in debug mode on all interfaces, port 5000
    app.run(debug=True, host="0.0.0.0", port=5000)