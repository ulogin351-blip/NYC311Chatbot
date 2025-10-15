from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    # Import the SQL Agent (renamed from sql_agent copy.py)
    from agent import SQLAgent as ChatAgent
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Make sure agent.py exists in the backend directory")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS for all routes and origins with enhanced configuration
CORS(app, 
     resources={r"/*": {
         "origins": ["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5173", "http://127.0.0.1:5174"],
         "methods": ["GET", "POST", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization", "Accept"],
         "expose_headers": ["Content-Type", "X-Total-Count"]
     }}, 
     supports_credentials=True)

# Global agent instance
agent = None

def initialize_agent():
    """Initialize the ChatAgent with error handling"""
    global agent
    try:
        logger.info("🔄 Initializing SQL Agent...")
        agent = ChatAgent()
        logger.info("✅ SQL Agent initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize SQL Agent: {e}")
        logger.error(f"📍 Full error: {traceback.format_exc()}")
        return False

# Initialize agent on startup - but don't exit if it fails
agent_initialized = initialize_agent()
if not agent_initialized:
    logger.error("Agent initialization failed - will initialize on first request")

@app.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint that processes user messages through the SQL Agent
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No JSON data provided",
                "success": False
            }), 400
        
        user_message = data.get("message", "").strip()
        conversation_history = data.get("conversation_history", [])
        
        if not user_message:
            return jsonify({
                "error": "Message cannot be empty",
                "success": False
            }), 400
        
        logger.info(f"🤖 Processing message: {user_message}")
        
        # Initialize agent if not already done
        if agent is None:
            if not initialize_agent():
                return jsonify({
                    "error": "Agent initialization failed. Check server logs.",
                    "success": False
                }), 500
        
        # Get response from agent with conversation history
        response_data = agent.get_response(user_message, conversation_history)
        
        # If the response is a string, convert to the expected format
        if isinstance(response_data, str):
            response_data = {
                "response": response_data,
                "requires_clarification": False
            }
        
        # Return successful response
        return jsonify({
            **response_data,
            "success": True,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        error_msg = f"Server error: {str(e)}"
        logger.error(f"❌ Error in chat endpoint: {error_msg}")
        logger.error(f"📍 Full traceback: {traceback.format_exc()}")
        
        return jsonify({
            "error": error_msg,
            "success": False,
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route("/", methods=["GET"])
def index():
    """
    Root endpoint with API information
    """
    return jsonify({
        "message": "SQL Agent Flask Backend",
        "version": "1.0.0",
        "endpoints": {
            "/chat": "POST - Send a message to the SQL Agent",
            "/": "GET - This endpoint"
        },
        "status": "running",
        "timestamp": datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested URL was not found on the server",
        "success": False
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "success": False
    }), 500

if __name__ == "__main__":
    # Force port 8000 regardless of environment variables
    host = "0.0.0.0"
    port = 8000
    debug = True
    
    # Override environment variables for clarity
    os.environ["FLASK_PORT"] = str(port)
    
    print("🚀 Starting SQL Agent Flask Backend")
    print("=" * 50)
    print(f"🌐 Server: http://localhost:{port}")
    print(f" Chat: POST to http://localhost:{port}/chat")
    print("=" * 50)
    
    try:
        app.run(debug=debug, host=host, port=port)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        if agent:
            agent.close()
    except Exception as e:
        print(f"❌ Server error: {e}")
        if agent:
            agent.close()