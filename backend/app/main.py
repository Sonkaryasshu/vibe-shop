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

    session_id = data.get("session_id")
    is_new_session = False
    if not session_id:
        session_id = str(uuid.uuid4())
        is_new_session = True
        current_app.logger.info(f"New session started: {session_id}")
    
    if is_new_session or session_id not in conversation_sessions:
        initial_vibe = data.get("vibe_description")
        if not initial_vibe:
            return jsonify({"error": "vibe_description is required to start a new conversation"}), 400
        
        session_state = {
            "vibe_description": initial_vibe,
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
        "vibe_description": session_state["vibe_description"],
        "current_filters": session_state.get("current_filters", {}),
        "user_response": data.get("user_response"),
        "last_question_text": session_state.get("last_question_text"),
        "questions_asked_history": session_state.get("questions_asked_history", [])
    }
    current_app.logger.info(f"Service payload for session {session_id}: {json.dumps(service_payload, indent=2)}")

    if not product_service_instance:
        current_app.logger.error("Product service not available for /converse.")
        return jsonify({"error": "Conversation service is currently unavailable"}), 503

    try:
        result = product_service_instance.converse(service_payload)
        
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
