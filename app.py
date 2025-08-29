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
from voice_service import VoiceService  # Import our fixed voice service

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Initialize OpenAI language model with GPT-4o-mini
llm = ChatOpenAI(model="gpt-4o-mini")

# Initialize OpenAI embeddings for vector search
embeddings = OpenAIEmbeddings()

# Initialize voice service with error handling
try:
    voice_service = VoiceService()
    print("Voice service initialized successfully!")
except Exception as e:
    print(f"Voice service initialization failed: {e}")
    voice_service = None

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

            Rules:
            - Do not answer any non-related cryptocurrency questions
            - Never be rude or disrespectful
            - Never use inappropriate language
            - Use the appropriate tools to find and answer relevant to a query
            - If the user responds in a certain language you respond in the same language or if they request for you to respond in a certain language, you respond in that language. Do not continue responding in the previous language, just respond in whatever language they texted in
            - Keep the responses not too long but impactful and make sure the response is informative.
            - Keep responses concise for voice synthesis - aim for 2-3 sentences max when voice is requested.
            - For voice responses, avoid special characters, excessive punctuation, and complex formatting
            
            If asked about non-crypto topics, respond in a respectful manner, explaining that the user's question is not a part of your function as a cryptocurrency assistant.
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
    verbose=True,
)

# Wrap agent with conversation history management using MongoDB
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
    try:
        session_id = request.json.get("session_id", "testing")
        query = request.json.get("query", "")
        include_voice = request.json.get("include_voice", False)
        
        if not query.strip():
            return jsonify({"error": "Query cannot be empty"}), 400
        
        # Invoke the agent with the user's query and session configuration
        response = agent_with_history.invoke(
            {"query": query},
            config={"configurable": {"session_id": session_id}},
        )
        
        result = {
            "output": response["output"],
            "session_id": session_id
        }
        
        # Generate voice if requested and voice service is available
        if include_voice and voice_service:
            try:
                # Clean the text for better voice synthesis
                voice_text = response["output"]
                # Remove markdown formatting and special characters
                import re
                voice_text = re.sub(r'\*\*(.*?)\*\*', r'\1', voice_text)  # Remove bold
                voice_text = re.sub(r'`(.*?)`', r'\1', voice_text)  # Remove code formatting
                voice_text = re.sub(r'[^\w\s.,!?-]', '', voice_text)  # Remove special chars
                
                audio_base64 = voice_service.generate_speech_base64(voice_text)
                if audio_base64:
                    result["audio"] = audio_base64
                    result["audio_format"] = "mp3"
                else:
                    result["voice_error"] = "Voice generation failed"
            except Exception as e:
                print(f"Voice generation error: {e}")
                result["voice_error"] = "Voice generation failed"
        elif include_voice and not voice_service:
            result["voice_error"] = "Voice service not available"
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"error": "An error occurred processing your request"}), 500

@app.route("/voice/test", methods=["GET"])
def test_voice():
    """Test endpoint for voice functionality."""
    if not voice_service:
        return jsonify({
            "success": False,
            "message": "Voice service not initialized"
        }), 500
    
    try:
        success = voice_service.test_voice()
        return jsonify({
            "success": success,
            "message": "Voice test completed. Check server logs and test_voice.mp3 file." if success else "Voice test failed. Check server logs."
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Voice test error: {str(e)}"
        }), 500

@app.route("/voice/debug", methods=["GET"])
def debug_voice():
    """Debug endpoint to test ElevenLabs connection."""
    try:
        # Check environment variables
        api_key = os.getenv("ELEVENLABS_API_KEY")
        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        
        if not api_key:
            return jsonify({
                "error": "ELEVENLABS_API_KEY not found in environment variables",
                "api_key_present": False
            }), 400
        
        # Test voice service initialization
        if not voice_service:
            return jsonify({
                "error": "Voice service failed to initialize",
                "api_key_present": bool(api_key),
                "api_key_length": len(api_key) if api_key else 0,
                "voice_id": voice_id
            }), 500
        
        # Get available voices (optional, might fail if quota exceeded)
        voices = None
        try:
            voices = voice_service.get_available_voices()
        except Exception as e:
            print(f"Could not fetch voices: {e}")
        
        # Test voice generation
        test_result = voice_service.test_voice("Testing Kryptonic AI voice service.")
        
        return jsonify({
            "api_key_present": True,
            "api_key_length": len(api_key),
            "voice_id": voice_service.voice_id,
            "test_successful": test_result,
            "voices_available": bool(voices),
            "voice_service_initialized": True,
            "message": "Voice service is working!" if test_result else "Voice service has issues. Check logs."
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "voice_service_initialized": voice_service is not None
        }), 500

@app.route("/voice/generate", methods=["POST"])
def generate_voice():
    """Standalone endpoint to generate voice from text."""
    if not voice_service:
        return jsonify({"error": "Voice service not available"}), 500
    
    try:
        data = request.get_json()
        text = data.get("text", "") if data else ""
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        if len(text) > 2500:
            return jsonify({"error": "Text too long (max 2500 characters)"}), 400
        
        audio_base64 = voice_service.generate_speech_base64(text)
        if audio_base64:
            return jsonify({
                "audio": audio_base64,
                "audio_format": "mp3",
                "success": True,
                "message": "Voice generated successfully"
            })
        else:
            return jsonify({"error": "Voice generation failed"}), 500
            
    except Exception as e:
        print(f"Voice generation error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "voice_service": voice_service is not None,
        "mongodb": bool(os.getenv("MONGODB_URL")),
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "elevenlabs": bool(os.getenv("ELEVENLABS_API_KEY"))
    })

if __name__ == "__main__":
    # Print startup information
    print("Starting Kryptonic AI...")
    print(f"OpenAI API Key: {'Present' if os.getenv('OPENAI_API_KEY') else 'Missing'}")
    print(f"MongoDB URL: {'Present' if os.getenv('MONGODB_URL') else 'Missing'}")
    print(f"ElevenLabs API Key: {'Present' if os.getenv('ELEVENLABS_API_KEY') else 'Missing'}")
    print(f"Voice Service: {'Initialized' if voice_service else 'Failed'}")
    
    app.run(debug=True, host="0.0.0.0", port=3030)