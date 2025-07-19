from flask import Blueprint, jsonify, request, current_app
from .services.product_service import product_service_instance
import uuid
import json

main_bp = Blueprint('main', __name__, url_prefix='/api')

conversation_sessions = {}

@main_bp.route('/')
def index():
    return jsonify({"message": "Welcome to the Apparel Recommendation API!"})

@main_bp.route('/search', methods=['POST'])
def search_products():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400
        
    query = data.get('query', '')
    top_k = data.get('top_k', 5)

    if not isinstance(top_k, int) or top_k <= 0:
        return jsonify({"error": "top_k must be a positive integer"}), 400

    if not query:
        return jsonify({"error": "Query parameter is missing"}), 400

    if not product_service_instance:
        current_app.logger.error("Product service not available.")
        return jsonify({"error": "Search service is currently unavailable"}), 503
    
    return jsonify({"message": "/search endpoint is for direct query. Use /converse for conversational search."})


@main_bp.route('/converse', methods=['POST'])
def converse_route():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    # Extract user input from any of these keys (user_response preferred, then vibe_description, then user_input)
    user_input = data.get("user_response") or data.get("vibe_description") or data.get("user_input") or ""
    session_id = data.get("session_id")
    is_new_session = False
    if not session_id:
        # Generate session ID using ProductService
        session_id = product_service_instance._generate_session_id()
        is_new_session = True
        current_app.logger.info(f"New session started: {session_id}")
    
    if is_new_session or session_id not in conversation_sessions:
        if not user_input:
            return jsonify({"error": "user_response or vibe_description is required to start a new conversation"}), 400
        
        session_state = {
            "vibe_description": user_input,
            "current_filters": data.get("current_filters", {}),
            "questions_asked_history": [],
            "last_question_text": None
        }
        conversation_sessions[session_id] = session_state
        current_app.logger.info(f"Initialized new session state for {session_id}: {session_state}")
    else:
        session_state = conversation_sessions[session_id]
        current_app.logger.info(f"Retrieved session state for {session_id}: {session_state}")

    service_payload = {
        "session_id": session_id,
        "vibe_description": session_state["vibe_description"],
        "current_filters": session_state.get("current_filters", {}),
        "user_response": user_input,
        "last_question_text": session_state.get("last_question_text"),
        "questions_asked_history": session_state.get("questions_asked_history", [])
    }

    if not product_service_instance:
        current_app.logger.error("Product service not available for /converse.")
        return jsonify({"error": "Conversation service is currently unavailable"}), 503

    try:
        result = product_service_instance.converse(service_payload)
        
        # Check if product service generated a new session ID (context switch)
        returned_session_id = result.get("session_id", session_id)
        
        if returned_session_id != session_id:
            # Context switch occurred - create new session state with new session ID
            current_app.logger.info(f"Context switch detected: {session_id} -> {returned_session_id}")
            
            # Create new session state for the new session ID
            new_session_state = {
                "vibe_description": user_input,  # New vibe from user input
                "current_filters": result.get("current_filters", {}),
                "questions_asked_history": result.get("questions_asked_history", []),
                "last_question_text": result.get("question_text_for_client")
            }
            
            conversation_sessions[returned_session_id] = new_session_state
            current_app.logger.info(f"Created new session state for {returned_session_id}: {new_session_state}")
            
            # Add context switch flag to result
            result["context_switched"] = True
            result["context_switch_message"] = "New conversation started - context switched to fresh query"
            
            # Use the new session ID for response
            session_id = returned_session_id
        else:
            # No context switch - update existing session state
            session_state["current_filters"] = result.get("current_filters", session_state["current_filters"])
            session_state["questions_asked_history"] = result.get("questions_asked_history", session_state["questions_asked_history"])
            session_state["last_question_text"] = result.get("question_text_for_client") 
            
            conversation_sessions[session_id] = session_state
            current_app.logger.info(f"Updated session state for {session_id} after service call: {session_state}")

        response_payload = {**result, "session_id": session_id}
        current_app.logger.info(f"Response payload for session {session_id}: {json.dumps(response_payload, indent=2)}")
        return jsonify(response_payload)

    except Exception as e:
        current_app.logger.error(f"Error in /converse endpoint for session {session_id}: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred during conversation."}), 500
