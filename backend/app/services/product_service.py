import pandas as pd
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
from google.genai import types
from anthropic import Anthropic
from openai import OpenAI
import uuid
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# Fix HuggingFace tokenizers warning in Flask/threading environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'chroma_db')
VIBE_EXAMPLES_PATH = os.path.join(DATA_DIR, 'vibe_to_attribute_examples.txt')
VALID_ATTRIBUTES_PATH = os.path.join(DATA_DIR, 'valid_attribute_values.json')

def _parse_llm_json_output(llm_text_response: str, logger=None) -> dict:
    if logger is None:
        logger = type('PrintLogger', (), {'error': print, 'info': print})

    try:
        json_start = llm_text_response.find('{')
        json_end = llm_text_response.rfind('}')
        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_str = llm_text_response[json_start : json_end+1]
            return json.loads(json_str)
        else:
            return json.loads(llm_text_response.strip())
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON output: '{llm_text_response}'. Error: {e}")
        return {}
    except Exception as e:
        logger.error(f"An unexpected error occurred during LLM JSON parsing: {e}")
        return {}


class ProductService:
    def __init__(self):
        self.vibe_examples_text_content = ""
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.gemini_client = None
        self.gemini_pro_model = None
        self.gemini_flash_model = None
        self.anthropic_client = None
        self.claude_model = None
        self.openai_client = None
        self.openai_model = None
        self.use_openai = False
        self.use_claude = False  # Toggle between Gemini and Claude
        self.MAX_FOLLOW_UP_QUESTIONS = 2
        self.valid_attribute_values = {}
        
        # Backend session management
        self.session_storage = {}  # Dictionary to store session data by session_id
        self.MAX_SESSIONS = 1000  # Maximum number of sessions to keep
        
        # Semantic query caching for performance optimization
        self.semantic_query_cache = {}  # Dictionary to cache semantic queries by normalized vibe
        self.MAX_CACHE_ENTRIES = 100  # Maximum number of cached queries

        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Successfully loaded SentenceTransformer model.")
        except Exception as e:
            print(f"Error loading SentenceTransformer model: {e}")
            self.embedding_model = None
        
        # Initialize Gemini
        try:
            google_api_key = os.getenv("GOOGLE_API_KEY")
            if google_api_key:
                self.gemini_client = genai.Client(api_key=google_api_key)
                self.gemini_pro_model = os.getenv("GEMINI_PRO_MODEL_NAME", "gemini-2.5-pro")
                self.gemini_flash_model = os.getenv("GEMINI_FLASH_MODEL_NAME", "gemini-2.5-flash")
                print(f"Successfully configured Gemini API with Pro model: {self.gemini_pro_model} and Flash model: {self.gemini_flash_model}")
            else:
                print("Warning: GOOGLE_API_KEY environment variable not found. Gemini LLM features will be disabled.")
                self.gemini_client = None
        except Exception as e:
            print(f"Error configuring Gemini API: {e}")
            self.gemini_client = None
        
        # Initialize Claude
        try:
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            if anthropic_api_key:
                self.anthropic_client = Anthropic(api_key=anthropic_api_key)
                self.claude_model = "claude-sonnet-4-20250514"
                self.use_claude = True
                print(f"Successfully configured Anthropic API with model: {self.claude_model}. Use Claude: {self.use_claude}")
            else:
                print("Warning: ANTHROPIC_API_KEY environment variable not found. Claude LLM features will be disabled.")
                self.anthropic_client = None
        except Exception as e:
            print(f"Error configuring Anthropic API: {e}")
            self.anthropic_client = None
        
        # Initialize OpenAI
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                self.openai_client = OpenAI(api_key=openai_api_key)
                self.openai_model = "gpt-4o"
                self.use_openai = True
                print(f"Successfully configured OpenAI API with model: {self.openai_model}. Use OpenAI: {self.use_openai}")
            else:
                print("Warning: OPENAI_API_KEY environment variable not found. OpenAI LLM features will be disabled.")
                self.openai_client = None
        except Exception as e:
            print(f"Error configuring OpenAI API: {e}")
            self.openai_client = None
        
        try:
            if not os.path.exists(DB_PATH):
                print(f"FATAL: ChromaDB database not found at {DB_PATH}. Please run `python backend/build_db.py` first.")
                self.chroma_client = None
                self.collection = None
            else:
                self.chroma_client = chromadb.PersistentClient(path=DB_PATH)
                self.collection = self.chroma_client.get_collection(name="apparel_products")
                print(f"Successfully connected to persistent ChromaDB at '{DB_PATH}' with {self.collection.count()} items.")
        except Exception as e:
            print(f"Error initializing ChromaDB from path '{DB_PATH}': {e}")
            self.chroma_client = None
            self.collection = None

        self._load_data()

    def _load_data(self):
        try:
            if os.path.exists(VIBE_EXAMPLES_PATH):
                with open(VIBE_EXAMPLES_PATH, 'r', encoding='utf-8') as f:
                    self.vibe_examples_text_content = f.read()
                print(f"Successfully loaded vibe examples from {VIBE_EXAMPLES_PATH}")
            else:
                print(f"Warning: Vibe examples file not found at {VIBE_EXAMPLES_PATH}.")
                self.vibe_examples_text_content = ""

            if os.path.exists(VALID_ATTRIBUTES_PATH):
                with open(VALID_ATTRIBUTES_PATH, 'r', encoding='utf-8') as f:
                    self.valid_attribute_values = json.load(f)
                print(f"Successfully loaded valid attributes from {VALID_ATTRIBUTES_PATH}")
            else:
                print(f"Warning: Valid attributes file not found at {VALID_ATTRIBUTES_PATH}. In-context filtering will be impaired.")
                self.valid_attribute_values = {}

        except Exception as e:
            print(f"Error loading data: {e}")
            if not self.vibe_examples_text_content:
                self.vibe_examples_text_content = ""
            if not hasattr(self, 'valid_attribute_values') or not self.valid_attribute_values:
                self.valid_attribute_values = {}

    def _call_llm(self, prompt: str, response_format: str = "text", thinking_budget: int = 0, use_pro_model: bool = False, context: str = "general", cacheable_prefix: str = None) -> str:
        """Call the configured LLM (Claude or Gemini) with the given prompt"""
        # Force Gemini Pro if explicitly requested
        if use_pro_model and self.gemini_client:
            try:
                start_time = time.time()
                config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=thinking_budget
                    )
                )
                if response_format == "json":
                    config.response_mime_type = "application/json"
                
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_pro_model,
                    contents=prompt,
                    config=config
                )
                end_time = time.time()
                print(f"{context} Gemini Pro call took {end_time - start_time:.2f} seconds.")
                return response.candidates[0].content.parts[0].text
            except Exception as e:
                print(f"Error calling Gemini Pro API: {e}. Falling back to Claude.")
                # Fall back to Claude if Gemini Pro fails
        
        if self.use_openai and self.openai_client:
            try:
                start_time = time.time()
                
                messages = [{"role": "user", "content": prompt}]
                
                request_params = {
                    "model": self.openai_model,
                    "messages": messages,
                    "max_tokens": 4096,
                }
                
                if response_format == "json":
                    request_params["response_format"] = {"type": "json_object"}
                    # Add instruction to prompt for JSON output
                    request_params["messages"] = [{"role": "user", "content": f"{prompt}\n\nPlease respond with valid JSON only."}]

                response = self.openai_client.chat.completions.create(**request_params)
                
                end_time = time.time()
                print(f"{context} OpenAI call took {end_time - start_time:.2f} seconds.")
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error calling OpenAI API: {e}. Falling back to other models.")
                # Fallback will happen by continuing execution

        if self.use_claude and self.anthropic_client:
            try:
                start_time = time.time()
                
                # Build messages with prompt caching if cacheable_prefix provided
                if cacheable_prefix:
                    # Structure: [cacheable system message] + [user message with specific query]
                    remaining_prompt = prompt[len(cacheable_prefix):].strip()
                    messages = [
                        {
                            "role": "user", 
                            "content": [
                                {
                                    "type": "text",
                                    "text": cacheable_prefix,
                                    "cache_control": {"type": "ephemeral"}
                                },
                                {
                                    "type": "text", 
                                    "text": remaining_prompt + ("\n\nPlease respond with valid JSON only." if response_format == "json" else "")
                                }
                            ]
                        }
                    ]
                else:
                    # Original single message format
                    if response_format == "json":
                        messages = [
                            {
                                "role": "user",
                                "content": f"{prompt}\n\nPlease respond with valid JSON only."
                            }
                        ]
                    else:
                        messages = [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                
                response = self.anthropic_client.messages.create(
                    model=self.claude_model,
                    max_tokens=5000,
                    messages=messages
                )
                end_time = time.time()
                print(f"{context} Claude call took {end_time - start_time:.2f} seconds.")
                return response.content[0].text
            except Exception as e:
                print(f"Error calling Claude API: {e}. Falling back to Gemini.")
                # Fall back to Gemini if Claude fails
        
        # Use Gemini (default or fallback)
        if self.gemini_client:
            try:
                start_time = time.time()
                config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=thinking_budget
                    )
                )
                if response_format == "json":
                    config.response_mime_type = "application/json"
                
                # Use Pro model if requested, otherwise use Flash
                model_to_use = self.gemini_pro_model if use_pro_model else self.gemini_flash_model
                
                response = self.gemini_client.models.generate_content(
                    model=model_to_use,
                    contents=prompt,
                    config=config
                )
                end_time = time.time()
                model_name = "Pro" if use_pro_model else "Flash"
                print(f"{context} Gemini {model_name} call took {end_time - start_time:.2f} seconds.")
                return response.candidates[0].content.parts[0].text
            except Exception as e:
                print(f"Error calling Gemini API: {e}")
                return ""
        
        print("No LLM client available (neither OpenAI, Claude nor Gemini).")
        return ""

    def _get_session_data(self, session_id: str) -> dict:
        """Get session data for a given session ID"""
        if session_id not in self.session_storage:
            # Clean up old sessions if we exceed the limit
            if len(self.session_storage) >= self.MAX_SESSIONS:
                # Remove oldest half of sessions
                sessions_to_remove = len(self.session_storage) // 2
                oldest_sessions = list(self.session_storage.keys())[:sessions_to_remove]
                for old_session in oldest_sessions:
                    del self.session_storage[old_session]
                print(f"Cleaned up {sessions_to_remove} old sessions. Current sessions: {len(self.session_storage)}")
            
            self.session_storage[session_id] = {
                "previous_vibe": None,
                "conversation_history": []
            }
        return self.session_storage[session_id]
    
    def _update_session_data(self, session_id: str, vibe: str, input_text: str):
        """Update session data with new vibe and input"""
        session_data = self._get_session_data(session_id)
        session_data["previous_vibe"] = vibe
        session_data["conversation_history"].append(input_text)
        # Keep only last 2 interactions to avoid memory bloat
        if len(session_data["conversation_history"]) > 2:
            session_data["conversation_history"] = session_data["conversation_history"][-2:]
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        return f"session_{uuid.uuid4().hex[:12]}"

    def _assess_shopping_intent(self, user_input: str, previous_vibe: str = None) -> dict:
        if not user_input:
            print("Empty input for shopping intent assessment. Defaulting to has_shopping_intent: False.")
            return {"has_shopping_intent": False, "suggested_reply_if_no_intent": "Hello! How can I help you find some apparel today?", "is_related_query": False}

        relatedness_section = ""
        if previous_vibe:
            relatedness_section = f"""
        
        ADDITIONAL TASK: Assess if this new query is related to the previous shopping context or completely different.
        Previous vibe/context: "{previous_vibe}"
        
        Determine if these are related or completely different:
        - RELATED: same category with refinements (e.g., "summer dresses" → "show full sleeves only")
        - DIFFERENT: completely different category/context (e.g., "summer dresses" → "work tops that go with pants")
        
        Include "is_related_query": boolean in your JSON response.
        """

        # Create cacheable prefix (static instructions + examples)
        cacheable_prefix = f"""You are a helpful assistant trying to understand if a user wants to shop for apparel.

INTENT DETECTION RULES:
- If the input indicates interest in shopping, browsing, or learning about apparel options (e.g., "looking for a dress", "summer clothes", "what do you have?", "tell me options", "show me products", "what categories", "effortless but polished", style descriptions), then the user has shopping intent.
- If the input is clearly unrelated to shopping for clothes (e.g., "what's the weather?", "who made you?", "how do I cook pasta?") OR is just a greeting (e.g., "hello", "hi", "hey"), then the user does not have shopping intent.
- When in doubt, assume the user has shopping intent.

OUTPUT FORMAT:
Output ONLY a JSON object with these keys:
1. "has_shopping_intent": boolean (true if shopping intent is present, false otherwise).
2. "suggested_reply_if_no_intent": string (If `has_shopping_intent` is false, provide a polite and helpful reply to guide the user towards stating their shopping needs. If `has_shopping_intent` is true, this should be null).
3. "is_related_query": boolean (true if related to previous context, false if completely different, null if no previous context).

EXAMPLES - SHOPPING INTENT:
"looking for a summer dress" → {{"has_shopping_intent": true, "suggested_reply_if_no_intent": null, "is_related_query": null}}
"what do you have?" → {{"has_shopping_intent": true, "suggested_reply_if_no_intent": null, "is_related_query": null}}
"tell me options" → {{"has_shopping_intent": true, "suggested_reply_if_no_intent": null, "is_related_query": null}}
"effortless but polished" → {{"has_shopping_intent": true, "suggested_reply_if_no_intent": null, "is_related_query": null}}
"casual weekend vibes" → {{"has_shopping_intent": true, "suggested_reply_if_no_intent": null, "is_related_query": null}}
"work clothes" → {{"has_shopping_intent": true, "suggested_reply_if_no_intent": null, "is_related_query": null}}
"party dress" → {{"has_shopping_intent": true, "suggested_reply_if_no_intent": null, "is_related_query": null}}
"something elegant" → {{"has_shopping_intent": true, "suggested_reply_if_no_intent": null, "is_related_query": null}}
"vacation outfits" → {{"has_shopping_intent": true, "suggested_reply_if_no_intent": null, "is_related_query": null}}
"show me tops" → {{"has_shopping_intent": true, "suggested_reply_if_no_intent": null, "is_related_query": null}}
"browse" → {{"has_shopping_intent": true, "suggested_reply_if_no_intent": null, "is_related_query": null}}
"shop" → {{"has_shopping_intent": true, "suggested_reply_if_no_intent": null, "is_related_query": null}}
"clothes" → {{"has_shopping_intent": true, "suggested_reply_if_no_intent": null, "is_related_query": null}}

EXAMPLES - NO SHOPPING INTENT:
"what's the weather today?" → {{"has_shopping_intent": false, "suggested_reply_if_no_intent": "I'm a shopping assistant. Are you looking for any clothing items?", "is_related_query": null}}
"hello" → {{"has_shopping_intent": false, "suggested_reply_if_no_intent": "Hello! What kind of vibe are you looking for today?", "is_related_query": null}}
"who made you?" → {{"has_shopping_intent": false, "suggested_reply_if_no_intent": "I'm an AI shopping assistant. What clothing items can I help you find?", "is_related_query": null}}
"how do I cook pasta?" → {{"has_shopping_intent": false, "suggested_reply_if_no_intent": "I specialize in fashion and apparel. What clothing are you shopping for?", "is_related_query": null}}

EXAMPLES - WITH PREVIOUS CONTEXT:
Previous: "summer dresses", New: "show full sleeves only" → {{"has_shopping_intent": true, "suggested_reply_if_no_intent": null, "is_related_query": true}}
Previous: "summer dresses", New: "work tops that go with pants" → {{"has_shopping_intent": true, "suggested_reply_if_no_intent": null, "is_related_query": false}}
{relatedness_section}

TASK:"""
        
        # Variable part
        variable_part = f"""
User's input: "{user_input}"

Analyze this input and provide the JSON response.

JSON:"""
        
        prompt = cacheable_prefix + variable_part
        try:
            response_text = self._call_llm(prompt, response_format="json", thinking_budget=0, context="intent_assessment", cacheable_prefix=cacheable_prefix)
            assessment_result = _parse_llm_json_output(response_text)
            if isinstance(assessment_result, dict) and "has_shopping_intent" in assessment_result:
                return {
                    "has_shopping_intent": assessment_result.get("has_shopping_intent", False),
                    "suggested_reply_if_no_intent": assessment_result.get("suggested_reply_if_no_intent"),
                    "is_related_query": assessment_result.get("is_related_query", None)
                }
            else:
                print(f"Failed to parse valid shopping intent assessment from LLM: {response_text}. Defaulting to has_shopping_intent: True.")
                return {"has_shopping_intent": True, "suggested_reply_if_no_intent": None, "is_related_query": None}
        except Exception as e:
            print(f"Error during _assess_shopping_intent: {e}. Defaulting to has_shopping_intent: True.")
            return {"has_shopping_intent": True, "suggested_reply_if_no_intent": None, "is_related_query": None}

    def _infer_attributes_from_vibe(self, vibe_description: str, previous_context: dict = None) -> dict:

        # Handle previous context
        previous_section = ""
        if previous_context:
            previous_vibe = previous_context.get("previous_vibe", "")
            previous_filters = previous_context.get("previous_filters", {})
            previous_types = previous_filters.get("attribute_types", {})
            
            # Remove attribute_types from filters for cleaner display
            clean_previous_filters = {k: v for k, v in previous_filters.items() if k != "attribute_types"}
            
            previous_section = f"""
        PREVIOUS CONTEXT:
        - Previous user input: "{previous_vibe}"
        - Previous filters extracted: {json.dumps(clean_previous_filters)}
        - Previous explicit/implicit types: {json.dumps(previous_types)}
        
        Your task is to update the filters based on the new user input below, considering:
        1. Keep explicit attributes from previous context unless directly contradicted
        2. Update or replace implicit attributes with new information
        3. Add new attributes from the current input
        4. Properly classify all final attributes as explicit or implicit
        """

        # Create cacheable prefix (static instructions + examples + valid values)
        cacheable_prefix = f"""You are a fashion expert helping update product search filters.

        Using the following examples of vibe-to-attribute mappings:
        --- VIBE EXAMPLES START ---
        {self.vibe_examples_text_content}
        --- VIBE EXAMPLES END ---

        Here are the ONLY VALID values for certain filterable attributes. When inferring these attributes, you MUST choose from these lists if you decide to include the attribute.
        CRITICAL: Values must match the case exactly as shown (e.g., "Work" not "work", "Sleeveless" not "sleeveless").
        If a suitable value is not in the list for an attribute, do NOT infer that attribute.
        --- VALID ATTRIBUTE VALUES START ---
        {json.dumps(self.valid_attribute_values, indent=2)}
        --- VALID ATTRIBUTE VALUES END ---

        Infer potential product attributes (like category, fit, fabric, color_or_print, occasion, sleeve_length, length, pant_type, price_range, size).
        Focus on attributes strongly implied by the vibe and supported by the examples AND constrained by the VALID ATTRIBUTE VALUES.
        For attributes listed in VALID ATTRIBUTE VALUES, only use values from the provided lists. For 'size', you can infer common sizes like S, M, L, XL, etc. or specific plus sizes.
        
        IMPORTANT: Pay special attention to specific attribute requirements:
        - If the vibe mentions "sleeveless", ensure you infer sleeve_length attribute with the exact value "Sleeveless" (match the VALID ATTRIBUTE VALUES)
        - If the vibe mentions "no black" or "no blue", create exclusion filters using "exclude_colors" key with the colors to exclude
        - If the vibe mentions "full sleeves" or "long sleeves", use the appropriate sleeve_length value from VALID ATTRIBUTE VALUES
        - If the vibe mentions multiple categories like "tops and dresses", include both in the category array
        
        For price:
        - If the vibe mentions a maximum (e.g., 'under $100', 'less than $100'), use 'price_max'.
        - If the vibe mentions a minimum (e.g., 'over $50', 'at least $50'), use 'price_min'.
        - If the vibe mentions a range (e.g., '$50 to $100', 'between $50 and $100'), extract both 'price_min' and 'price_max'.
        If the vibe mentions "plus size", "plus sized", "curvy", or similar terms, you should infer the 'size' attribute to include larger sizes such as ["XL", "XXL", "1X", "2X"].
        
        CRITICAL: Be extremely conservative with restrictive attributes (fit, fabric, occasion) for broad vibes. 
        For broad vibes like "date night", "party", "brunch", "casual", avoid inferring restrictive attributes unless explicitly mentioned.
        For example, "date night" should NOT infer specific fit, fabric, or occasion unless the user specifically mentions these.
        Only infer attributes that are explicitly mentioned or absolutely necessary for the vibe.
        If an attribute is not strongly implied or a valid value cannot be found, do not include it.
        IMPORTANT NEW REQUIREMENT: For each attribute you infer, also determine if it's "explicit" or "implicit":
        - EXPLICIT: User directly mentioned this attribute (e.g., "red dress", "work clothes", "size M", "under $100", "sleeveless", "linen tops")
        - IMPLICIT: You inferred this attribute based on the general vibe (e.g., inferring "Cotton" for "summer casual", inferring "Party" occasion for "date night")
        
        Output your answer as a JSON object with two sections:
        1. "attributes": regular attribute values
        2. "attribute_types": mapping of each attribute to "explicit" or "implicit"
        
        Examples using real attribute values from our catalog:
        
        Vibe: "something flowy for date night"
        {{"attributes": {{"fit": ["Flowy"], "occasion": ["Evening", "Party"], "category": ["dress", "top", "skirt"]}}, "attribute_types": {{"fit": "explicit", "occasion": "explicit", "category": "implicit"}}}}
        
        Vibe: "professional but comfortable bottoms"  
        {{"attributes": {{"category": ["pants"], "occasion": ["Work"], "fit": ["Relaxed", "Tailored"]}}, "attribute_types": {{"category": "implicit", "occasion": "explicit", "fit": "explicit"}}}}
        
        Vibe: "bright summer vibes up to knee length"
        {{"attributes": {{"fabric": ["Linen", "Cotton"], "length": ["Mini", "Short", "Knee length"], "color_or_print": ["Sunflower yellow", "Sunshine yellow", "Coral stripe", "Pastel yellow"]}}, "attribute_types": {{"fabric": "implicit", "length": "explicit", "color_or_print": "implicit"}}}}
        
        Vibe: "effortless but polished"
        {{"attributes": {{"fit": ["Relaxed", "Tailored"], "fabric": ["Tencel", "Modal jersey", "Silk"], "occasion": ["Work", "Everyday"]}}, "attribute_types": {{"fit": "implicit", "fabric": "implicit", "occasion": "implicit"}}}}
        
        Vibe: "brunch outfit - something cute and comfy"  
        {{"attributes": {{"occasion": ["Everyday"], "fit": ["Relaxed", "Flowy"], "category": ["dress", "top"], "fabric": ["Cotton", "Modal jersey", "Rayon"]}}, "attribute_types": {{"occasion": "implicit", "fit": "explicit", "category": "implicit", "fabric": "implicit"}}}}
        
        Vibe: "vacation ready - breathable and loose"
        {{"attributes": {{"occasion": ["Vacation"], "fit": ["Relaxed", "Flowy"], "fabric": ["Linen", "Cotton gauze", "Viscose voile"], "category": ["dress", "top", "pants"]}}, "attribute_types": {{"occasion": "explicit", "fit": "explicit", "fabric": "explicit", "category": "implicit"}}}}
        
        Vibe: "night out dancing - want to move freely"
        {{"attributes": {{"occasion": ["Party", "Evening"], "fit": ["Stretch to fit", "Body hugging"], "fabric": ["Stretch denim", "Ribbed jersey", "Crepe"]}}, "attribute_types": {{"occasion": "explicit", "fit": "implicit", "fabric": "implicit"}}}}
        
        Vibe: "client meeting tomorrow - navy or black preferred"
        {{"attributes": {{"occasion": ["Work"], "color_or_print": ["Midnight navy", "Jet black", "Charcoal"], "fit": ["Tailored"]}}, "attribute_types": {{"occasion": "implicit", "color_or_print": "explicit", "fit": "implicit"}}}}
        
        Vibe: "work blouses no sleeves"
        {{"attributes": {{"category": ["top"], "occasion": ["Work"], "sleeve_length": "Sleeveless"}}, "attribute_types": {{"category": "implicit", "occasion": "explicit", "sleeve_length": "explicit"}}}}
        
        Vibe: "brunch with friends"
        {{"attributes": {{"occasion": ["Everyday"], "category": ["dress", "top"], "fit": ["Relaxed", "Flowy"]}}, "attribute_types": {{"occasion": "implicit", "category": "implicit", "fit": "implicit"}}}}
        
        Vibe: "party dress red under $100"
        {{"attributes": {{"category": ["dress"], "occasion": ["Party"], "color_or_print": ["Red"], "price_max": 100}}, "attribute_types": {{"category": "explicit", "occasion": "implicit", "color_or_print": "explicit", "price_max": "explicit"}}}}
        
        Vibe: "shimmery metallic fabric for party"
        {{"attributes": {{"fabric": ["Lamé", "Sequined mesh"], "occasion": ["Party"]}}, "attribute_types": {{"fabric": "explicit", "occasion": "explicit"}}}}
        
        If no attributes can be confidently inferred, output: {{"attributes": {{}}, "attribute_types": {{}}}}.

        Task Instructions:"""
        
        # Variable part (changes per request)
        variable_part = f"""
        {previous_section}
        Current user input: "{vibe_description}"
        
        Based on the above instructions and examples, analyze the current user input and output the JSON response.
        JSON:
        """
        
        prompt = cacheable_prefix + variable_part
        try:
            response_text = self._call_llm(prompt, response_format="json", thinking_budget=1024, context="attribute_inference", cacheable_prefix=cacheable_prefix)
            llm_response = _parse_llm_json_output(response_text)

            # Expect new format with attributes and attribute_types
            inferred_attributes = llm_response.get("attributes", {})
            attribute_types = llm_response.get("attribute_types", {})

            if inferred_attributes and self.valid_attribute_values:
                validated_attributes = {}
                validated_attribute_types = {}
                
                for key, value in inferred_attributes.items():
                    if key in self.valid_attribute_values:
                        valid_options_for_key = self.valid_attribute_values[key]
                        if isinstance(value, list):
                            cleaned_values = [v for v in value if v in valid_options_for_key]
                            if cleaned_values:
                                validated_attributes[key] = cleaned_values
                                validated_attribute_types[key] = attribute_types.get(key, "implicit")
                        elif isinstance(value, str):
                            if value in valid_options_for_key:
                                validated_attributes[key] = value
                                validated_attribute_types[key] = attribute_types.get(key, "implicit")
                    else:
                        validated_attributes[key] = value
                        validated_attribute_types[key] = attribute_types.get(key, "implicit")
                
                # Add attribute_types to the result
                if validated_attribute_types:
                    validated_attributes["attribute_types"] = validated_attribute_types
                
                print(f"🎯 INFERRED ATTRIBUTES: {inferred_attributes}")
                print(f"📋 EXPLICIT/IMPLICIT TYPES: {attribute_types}")
                print(f"✅ VALIDATED FILTERS: {validated_attributes}")
                return validated_attributes
            else:
                result = {"attribute_types": attribute_types} if attribute_types else {}
                return result

        except Exception as e:
            print(f"Error inferring attributes from vibe: {e}")
            return {}

    def _parse_user_answer_and_update_filters(self, last_question_text: str, user_answer: str, current_filters: dict) -> dict:

        filters_for_prompt = {k: v for k, v in current_filters.items() if k != "vibe_inferred"}

        prompt = f"""
        You are a helpful assistant processing a user's preferences for apparel.
        Current known preferences: {json.dumps(filters_for_prompt)}
        Valid attribute values: {json.dumps(self.valid_attribute_values)}
        The user was asked: "{last_question_text}"
        The user replied: "{user_answer}"

        Based on the user's reply, identify what attributes from the preferences should be updated, added, or removed.
        
        IMPORTANT: Extract ANY relevant attribute information from the user's response, even if it doesn't directly answer the original question. 
        For example, if asked about "fit" but user mentions "size", extract the size information.
        
        Output ONLY a JSON object containing these changes.
        - To add or update an attribute, include its new value (e.g., {{"price_max": 50}}, {{"size": ["S"]}}).
        - If the user's reply indicates a preference for an attribute should be cleared or reset (e.g., they say "any size is fine" or "no budget limit"), output that attribute with a `null` value (e.g., {{"size": null}}).
        - Include ALL attributes directly addressed or modified by the user's current reply. Do not include unchanged attributes from 'Current known preferences'.
        - If the user's answer contains no relevant attribute information, return an empty JSON object {{}}.
        - If the user is asking a clarifying question instead of providing preference information, return: {{"clarification_answer": "your helpful answer to their question"}}

        For example:
        - If Current preferences are {{"category": "top"}} and user was asked "Budget?" and replied "under $50", your JSON output should be: {{"price_max": 50}}
        - If Current preferences are {{"price_max": 100}} and user was asked "Size?" and replied "S or M", your JSON output should be: {{"size": ["S", "M"]}}
        - If Current preferences are {{"size": "S"}} and user was asked "Size?" and replied "Actually, any size works", your JSON output should be: {{"size": null}}
        - If user was asked "What fit are you looking for?" and replied "i need small size only", your JSON output should be: {{"size": ["S"]}}
        - If user was asked "What occasion?" and replied "casual wear, medium budget around $75", your JSON output should be: {{"occasion": "casual", "price_max": 75}}
        - If the question was "Any must-haves like sleeveless, budget range or size to keep in mind?" and the user replied "Want sleeveless, keep under $100, both S and M work", your JSON output should be:
          {{"sleeve_length": "sleeveless", "price_max": 100, "size": ["S", "M"]}}
        - If user was asked "What category?" and replied "what categories do you have?", your JSON output should be: {{"clarification_answer": "I have these categories available: dress, top, pants, skirt. Which one interests you for your effortless but polished look?"}}
        
        Ensure attribute keys in your JSON output are standard (e.g., price_min, price_max, category, size, fit, fabric, color_or_print, occasion, sleeve_length, length, pant_type).
        JSON:
        """
        try:
            response_text = self._call_llm(prompt, response_format="json", thinking_budget=0, context="follow_up_answer_parsing")
            llm_suggested_changes = _parse_llm_json_output(response_text)
            if isinstance(llm_suggested_changes, dict):
                # Check if this is a clarification answer
                if "clarification_answer" in llm_suggested_changes:
                    # Return special marker for clarification
                    return {"__clarification_answer__": llm_suggested_changes["clarification_answer"]}
                
                # Normal filter updates
                if llm_suggested_changes:
                    new_filters = current_filters.copy()
                    for key, value in llm_suggested_changes.items():
                        if value is None:
                            if key in new_filters:
                                del new_filters[key]
                        else:
                            new_filters[key] = value
                    return new_filters
            return current_filters
        except Exception as e:
            print(f"Error parsing user answer: {e}")
            return current_filters

    def _build_chroma_where_clause(self, filters: dict) -> Optional[dict]:
        where_conditions = []
        processed_keys = set()

        if "price_min" in filters and "price_max" in filters and filters["price_min"] is not None and filters["price_max"] is not None:
            where_conditions.append({"$and": [{"price": {"$gte": float(filters["price_min"])}}, {"price": {"$lte": float(filters["price_max"])}}]})
        elif "price_min" in filters and filters["price_min"] is not None:
            where_conditions.append({"price": {"$gte": float(filters["price_min"])}})
        elif "price_max" in filters and filters["price_max"] is not None:
            where_conditions.append({"price": {"$lte": float(filters["price_max"])}})
        processed_keys.update(["price_min", "price_max", "budget", "vibe_inferred", "exclude_colors", "attribute_types"])


        for key, value in filters.items():
            if key in processed_keys or value is None or value == "" or (isinstance(value, list) and not value):
                continue
            
            if key == "size":
                continue

            if isinstance(value, list):
                if len(value) == 1:
                     where_conditions.append({key: {"$eq": str(value[0])}})
                elif len(value) > 1:
                    or_clauses = [{key: {"$eq": str(v_item)}} for v_item in value]
                    where_conditions.append({"$or": or_clauses})
            else:
                where_conditions.append({key: {"$eq": str(value) if not isinstance(value, (int, float, bool)) else value}})
        
        if not where_conditions:
            return None
        if len(where_conditions) == 1:
            return where_conditions[0]
        return {"$and": where_conditions}

    def _apply_python_filters(self, products: list, filters: dict) -> list:
        filtered_products = products

        user_sizes_str = filters.get("size")
        if user_sizes_str:
            user_s_list = []
            if isinstance(user_sizes_str, list):
                user_s_list = [s.strip().upper() for s in user_sizes_str]
            elif isinstance(user_sizes_str, str):
                user_s_list = [s.strip().upper() for s in user_sizes_str.split(',')]
            
            if user_s_list:
                temp_products = []
                for product in filtered_products:
                    available_sizes_product = product.get("available_sizes", "")
                    if available_sizes_product and isinstance(available_sizes_product, str):
                        product_s_list = {s.strip().upper() for s in available_sizes_product.split(',')}
                        if any(size_filter in product_s_list for size_filter in user_s_list):
                            temp_products.append(product)
                filtered_products = temp_products
        
        # Handle color exclusions with string matching
        exclude_colors = filters.get("exclude_colors")
        if exclude_colors:
            exclude_colors_list = []
            if isinstance(exclude_colors, list):
                exclude_colors_list = [color.strip().lower() for color in exclude_colors]
            elif isinstance(exclude_colors, str):
                exclude_colors_list = [exclude_colors.strip().lower()]
            
            if exclude_colors_list:
                temp_products = []
                for product in filtered_products:
                    color_or_print = product.get("color_or_print", "").lower()
                    # Check if any excluded color appears in the color_or_print field
                    if not any(excluded_color in color_or_print for excluded_color in exclude_colors_list):
                        temp_products.append(product)
                filtered_products = temp_products
        
        price_min = filters.get("price_min")
        price_max = filters.get("price_max")

        if price_min is not None:
            filtered_products = [p for p in filtered_products if p.get("price", float('inf')) >= float(price_min)]
        if price_max is not None:
            filtered_products = [p for p in filtered_products if p.get("price", float('-inf')) <= float(price_max)]

        return filtered_products

    def _refine_query_based_on_vibe(self, vibe_description: str) -> str:
        # Check cache first
        cache_key = self._normalize_vibe_for_cache(vibe_description)
        if cache_key in self.semantic_query_cache:
            cached_query = self.semantic_query_cache[cache_key]
            print(f"🎯 CACHE HIT: Using cached semantic query for '{vibe_description}' (key: '{cache_key}')")
            return cached_query
        
        print(f"📝 CACHE MISS: Generating new semantic query for '{vibe_description}' (key: '{cache_key}')")
        
        # Include available attribute values in the prompt
        available_values_text = ""
        if hasattr(self, 'valid_attribute_values') and self.valid_attribute_values:
            available_values_text = f"\n--- AVAILABLE ATTRIBUTE VALUES ---\n{json.dumps(self.valid_attribute_values, indent=2)}\n--- END AVAILABLE VALUES ---\n"
        
        # Create cacheable prefix (static content)
        cacheable_parts = [
            "You are a fashion assistant. Your task is to translate a user's desired \"vibe\" into a descriptive textual query that can be used for semantic search of apparel.",
            "Use the following examples of how vibes map to product attributes as a guide:",
            "--- VIBE EXAMPLES START ---",
            self.vibe_examples_text_content,
            "--- VIBE EXAMPLES END ---",
            available_values_text,
            "\nBased on the user's vibe, the provided examples, and the available attribute values above, generate a descriptive textual query that captures the essence of this style.",
            "Be INCLUSIVE - when generating descriptions, include multiple variations that could match the vibe. Use the available attribute values as reference but don't limit yourself to only those exact terms.",
            "For elegant styles: include both sophisticated formal pieces AND elevated casual pieces, various luxurious fabrics (sequins, satin, silk, velvet, etc.), different elegant occasions (parties, dinners, events, evening wear).",
            "For casual styles: include comfortable fits, everyday fabrics, versatile pieces suitable for daily wear.",
            "Focus on the overall aesthetic feeling and style characteristics while being broad enough to match diverse interpretations of the vibe.",
            "\nFor vague inputs like 'buy', 'shop', or 'clothes', generate a broad description covering popular versatile pieces like 'casual comfortable clothing, everyday wear pieces, versatile tops and bottoms in neutral colors, suitable for multiple occasions'.",
            "\nALWAYS generate a valid search description. Never refuse or explain why you cannot help. Output only the detailed textual description for semantic search."
        ]
        cacheable_prefix = "\n".join(cacheable_parts)
        
        # Variable part (changes per request)
        variable_part = f"\n\nUser's desired vibe: \"{vibe_description}\"\n\nDetailed Description:"
        
        prompt = cacheable_prefix + variable_part
        
        refined_query = None
        try:
            response_text = self._call_llm(prompt, response_format="text", thinking_budget=256, context="semantic_query", cacheable_prefix=cacheable_prefix)
            if response_text:
                refined_query = response_text.strip()
                print(f"LLM refined query: '{refined_query}'")
            else:
                print("LLM response was empty. Falling back.")
        except Exception as e:
            print(f"LLM API call failed: {e}. Falling back.")
        
        if not refined_query:
            refined_query = vibe_description
            print(f"Falling back to original vibe description: '{refined_query}'")
        
        # Cache the result
        self._cache_semantic_query(cache_key, refined_query)
        
        return refined_query
    
    def _cache_semantic_query(self, cache_key: str, semantic_query: str):
        """Cache a semantic query with LRU-like cleanup"""
        # Simple cache size management - remove oldest entries when limit exceeded
        if len(self.semantic_query_cache) >= self.MAX_CACHE_ENTRIES:
            # Remove first (oldest) entry - simple FIFO cleanup
            oldest_key = next(iter(self.semantic_query_cache))
            del self.semantic_query_cache[oldest_key]
            print(f"💾 CACHE: Evicted oldest entry '{oldest_key}' to make room")
        
        self.semantic_query_cache[cache_key] = semantic_query
        print(f"💾 CACHE: Stored semantic query for key '{cache_key}' (cache size: {len(self.semantic_query_cache)})")

    def _normalize_vibe_for_cache(self, vibe_description: str) -> str:
        """Normalize vibe description for cache key generation"""
        import re
        # Convert to lowercase and remove extra whitespace
        normalized = vibe_description.lower().strip()
        # Remove common stopwords that don't affect semantic meaning
        stopwords = {'and', 'or', 'the', 'a', 'an', 'for', 'with', 'in', 'on', 'at', 'to', 'from', 'that', 'which', 'some', 'any'}
        words = normalized.split()
        words = [word for word in words if word not in stopwords]
        # Sort words to handle different orderings of same concepts
        words.sort()
        return ' '.join(words)

    def _parallel_attribute_and_semantic_calls(self, vibe_for_attributes, vibe_for_semantic, previous_context=None):
        """Run attribute inference and semantic query generation in parallel"""
        import time
        start_time = time.time()
        print(f"🚀 PARALLEL START: Starting parallel LLM calls")
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks concurrently
            if previous_context:
                attribute_future = executor.submit(self._infer_attributes_from_vibe, vibe_for_attributes, previous_context)
            else:
                attribute_future = executor.submit(self._infer_attributes_from_vibe, vibe_for_attributes)
            semantic_future = executor.submit(self._refine_query_based_on_vibe, vibe_for_semantic)
            
            print(f"⏱️  PARALLEL: Both tasks submitted, waiting for results...")
            
            # Get results (this waits for both to complete)
            attributes_result = attribute_future.result()
            print(f"✅ PARALLEL: Attribute inference completed")
            semantic_result = semantic_future.result()
            print(f"✅ PARALLEL: Semantic query completed")
            
            end_time = time.time()
            total_time = end_time - start_time
            print(f"🏁 PARALLEL COMPLETE: Total parallel execution time: {total_time:.2f} seconds")
            
            return attributes_result, semantic_result

    def _generate_justification_with_followup(self, vibe_description: str, products: list, current_filters: dict, 
                                            questions_asked_history: list, search_relaxed: bool = False) -> str:

        if not products:
            if search_relaxed:
                return "We relaxed filters to find more options, but no products matched. Try a different style or adjust preferences."
            return "No products found to justify based on the current criteria."

        product_details_list = []
        for i, p_dict in enumerate(products):
            name = p_dict.get('name', 'N/A')
            category = p_dict.get('category', 'N/A')
            fit = p_dict.get('fit', '')
            fabric = p_dict.get('fabric', '')
            color_or_print = p_dict.get('color_or_print', '')
            
            summary = f"Product {i+1}: {name} ({category}). "
            features = [f for f in [fit, fabric, color_or_print] if f]
            if features:
                summary += f"Key features: {', '.join(features)}."
            product_details_list.append(summary.strip())
        
        product_details_string = "\n".join(product_details_list)

        filter_summary_parts = []
        for key, value in current_filters.items():
            if not value or key in ["budget"]:
                continue
            if key == "price_max" and value is not None: filter_summary_parts.append(f"under ${value}")
            elif key == "price_min" and value is not None: filter_summary_parts.append(f"over ${value}")
            elif key == "size" and value: filter_summary_parts.append(f"size(s) {value if isinstance(value, str) else ', '.join(value)}")
            elif isinstance(value, list): filter_summary_parts.append(f"{key.replace('_', ' ')}: {', '.join(map(str,value))}")
            else: filter_summary_parts.append(f"{key.replace('_', ' ')}: {value}")
        filter_summary = "; ".join(filter_summary_parts)

        relaxation_instruction = ""
        if search_relaxed:
            relaxation_instruction = "\n\nIMPORTANT: Start your justification by mentioning that filters were relaxed to find these options (e.g., 'We relaxed your search to find...' or 'After broadening criteria...')."

        # Determine if follow-up question should be included
        follow_up_section = ""
        if len(questions_asked_history) < self.MAX_FOLLOW_UP_QUESTIONS:
            # Check what key attributes are missing
            missing_attrs = []
            if not current_filters.get("size"):
                missing_attrs.append("size")
            if not current_filters.get("price_max") and not current_filters.get("price_min"):
                missing_attrs.append("budget")
            if not current_filters.get("fit"):
                missing_attrs.append("fit preference")
            if not current_filters.get("occasion") and "occasion" not in [attr for attr in current_filters.get("attribute_types", {}) if current_filters["attribute_types"].get(attr) == "explicit"]:
                missing_attrs.append("occasion")
            
            if missing_attrs:
                follow_up_section = f"""

OPTIONAL FOLLOW-UP: If there are still important missing attributes that would improve recommendations, you may include a natural follow-up question at the end. 
Missing attributes that could help: {', '.join(missing_attrs)}
Questions already asked: {questions_asked_history}
Max {self.MAX_FOLLOW_UP_QUESTIONS} follow-ups total.

If including a follow-up, format it naturally at the end like: "To refine this further, what size are you looking for?" or "Do you have a budget range in mind?"
Only include if genuinely helpful - don't force it."""

        prompt = f"""The user expressed a desire for products matching the vibe: "{vibe_description}".
Additionally, they specified the following preferences: {filter_summary if filter_summary else "no specific additional preferences"}.

Based on this, we have recommended the following products:
--- RECOMMENDED PRODUCTS START ---
{product_details_string}
--- RECOMMENDED PRODUCTS END ---

Please provide a VERY brief, concise justification (1-2 sentences maximum) explaining why these products match their vibe.
Focus on the key attributes that align with the vibe. Be conversational and direct.

Example format: "These picks capture 'effortless' through relaxed fabrics and 'polished' with refined tones and tailored cuts—like the structured Mustard Muse top."

IMPORTANT: Keep the justification under 30 words. Be specific about how the products match the vibe, not generic descriptions.{relaxation_instruction}{follow_up_section}

Response:
"""
        try:
            response_text = self._call_llm(prompt, response_format="text", thinking_budget=512, context="justification")
            return response_text.strip() if response_text else "We found some great products for you! Their styles and features should match your vibe."
        except Exception as e:
            print(f"Error generating justification: {e}")
            return "We found some great products for you! Their styles and features should match your vibe."

    def _initialize_conversation_context(self, session_payload: dict) -> dict:
        """Initialize and validate conversation context"""
        session_id = session_payload.get("session_id") or self._generate_session_id()
        vibe = session_payload.get("vibe_description")
        current_filters = session_payload.get("current_filters", {})
        user_response = session_payload.get("user_response")
        last_question_text = session_payload.get("last_question_text")
        questions_asked_history = session_payload.get("questions_asked_history", [])
        
        # Get previous vibe from backend session storage
        session_data = self._get_session_data(session_id)
        previous_vibe = session_data.get("previous_vibe")
        
        return {
            "session_id": session_id,
            "vibe": vibe,
            "current_filters": current_filters,
            "user_response": user_response,
            "last_question_text": last_question_text,
            "questions_asked_history": questions_asked_history,
            "previous_vibe": previous_vibe,
            "is_follow_up": bool(user_response and last_question_text)
        }

    def _handle_follow_up_response(self, context: dict) -> dict:
        """Handle user responses to follow-up questions"""
        final_response = self._create_base_response(context)
        
        print(f"User is responding to follow-up question: '{context['last_question_text']}'. Checking shopping intent.")
        input_to_assess = context["user_response"]
        
        # Check if follow-up response is related to previous context or a fresh query
        print(f"📜 PREVIOUS: '{context['previous_vibe']}'")
        print(f"💬 CURRENT: '{context['user_response']}'")
        
        intent_assessment = self._assess_shopping_intent(context["user_response"], context["previous_vibe"])
        print(f"DEVLOG: Follow-up LLM assessment result: {intent_assessment}")
        
        has_shopping_intent = intent_assessment.get("has_shopping_intent", True)
        suggested_reply_if_no_intent = intent_assessment.get("suggested_reply_if_no_intent")
        is_related_query = intent_assessment.get("is_related_query", True)
        
        # Check if follow-up response has no shopping intent
        if not has_shopping_intent:
            print(f"Follow-up response '{context['user_response']}' deemed to have no shopping intent.")
            final_response["justification"] = suggested_reply_if_no_intent or "How can I help you find some apparel today?"
            final_response["products"] = []
            
            # Check for context switch even for non-shopping follow-up responses
            if not is_related_query:
                # Fresh query with no shopping intent - generate new session ID
                session_id = self._generate_session_id()
                print(f"Generated new session ID for fresh non-shopping follow-up: {session_id}")
                final_response["session_id"] = session_id
                context["session_id"] = session_id
            
            # Update session storage even for non-shopping follow-up inputs
            self._update_session_data(context["session_id"], context["vibe"] or input_to_assess, input_to_assess)
            print(f"DEVLOG: Updated session storage for non-shopping follow-up - vibe: '{context['vibe'] or input_to_assess}', input: '{input_to_assess}'")
            
            return final_response
        
        print(f"Follow-up input '{input_to_assess}' has shopping intent. Proceeding with query processing.")
        # Continue with query processing
        return self._process_shopping_query(context, input_to_assess, is_related_query)

    def _handle_initial_query(self, context: dict) -> dict:
        """Handle initial queries (not follow-up responses)"""
        final_response = self._create_base_response(context)
        
        # Determine input to assess
        input_to_assess = context["user_response"] if context["user_response"] else context["vibe"]
        
        if not input_to_assess:
            final_response["justification"] = "Hello! How can I help you find some apparel today?"
            final_response["products"] = []
            final_response["session_id"] = context["session_id"]
            return final_response

        # Get previous vibe for relatedness assessment
        print(f"📜 PREVIOUS: '{context['previous_vibe']}'")
        print(f"💬 CURRENT: '{input_to_assess}'")
        intent_assessment = self._assess_shopping_intent(input_to_assess, context["previous_vibe"])
        print(f"DEVLOG: LLM intent assessment result: {intent_assessment}")
        
        has_shopping_intent = intent_assessment.get("has_shopping_intent", True)
        suggested_reply_if_no_intent = intent_assessment.get("suggested_reply_if_no_intent")
        is_related_query = intent_assessment.get("is_related_query", None)

        if not has_shopping_intent:
            print(f"Input '{input_to_assess}' deemed to have no shopping intent.")
            final_response["justification"] = suggested_reply_if_no_intent or "How can I help you find some apparel today?"
            final_response["products"] = []
            
            # Check for context switch even for non-shopping intent responses
            if not is_related_query:
                # Fresh query with no shopping intent - generate new session ID
                session_id = self._generate_session_id()
                print(f"Generated new session ID for fresh non-shopping query: {session_id}")
                final_response["session_id"] = session_id
                context["session_id"] = session_id
            
            # Update session storage even for non-shopping inputs
            self._update_session_data(context["session_id"], context["vibe"] or input_to_assess, input_to_assess)
            print(f"DEVLOG: Updated session storage for non-shopping input - vibe: '{context['vibe'] or input_to_assess}', input: '{input_to_assess}'")
            
            return final_response
        
        print(f"Input '{input_to_assess}' has shopping intent. Proceeding with product logic.")
        return self._process_shopping_query(context, input_to_assess, is_related_query)

    def _create_base_response(self, context: dict) -> dict:
        """Create base response structure"""
        return {
            "session_id": context["session_id"],
            "current_filters": dict(context["current_filters"]),
            "questions_asked_history": list(context["questions_asked_history"]),
            "products": None,
            "justification": None
        }

    def _process_shopping_query(self, context: dict, input_to_assess: str, is_related_query: bool) -> dict:
        """Process shopping queries and execute search"""
        final_response = self._create_base_response(context)
        current_filters = context["current_filters"]
        questions_asked_history = context["questions_asked_history"]
        vibe = context["vibe"]
        
        # Handle query context management - related vs fresh queries
        if is_related_query is not None:
            print(f"Query relatedness assessment: {'related' if is_related_query else 'fresh'}")
            if not is_related_query:
                # Fresh query - reset filters and questions history, generate new session ID
                print("Fresh query detected - resetting context and generating new session ID.")
                current_filters = {}
                questions_asked_history = []
                # Generate new session ID for fresh query
                session_id = self._generate_session_id()
                print(f"Generated new session ID for fresh query: {session_id}")
                context["session_id"] = session_id
                final_response["session_id"] = session_id
                # Update vibe to the new query (use the current input as the new vibe)
                vibe = input_to_assess
                print(f"Updated vibe for fresh query: '{vibe}'")
                
                # Process fresh query
                processing_result = self._process_fresh_query(vibe, current_filters)
                current_filters = processing_result["current_filters"]
            else:
                print("Related query detected - retaining context and combining vibe.")
                # Process related query
                processing_result = self._process_related_query(context, input_to_assess)
                vibe = processing_result["vibe"]
                current_filters = processing_result["current_filters"]
        else:
            # When relatedness cannot be determined (no previous context), treat as fresh query
            print("No previous context available - treating as fresh query.")
            current_filters = {}
            questions_asked_history = []
            # Generate new session ID for fresh query (no previous context)
            session_id = self._generate_session_id()
            print(f"Generated new session ID for fresh query (no previous context): {session_id}")
            context["session_id"] = session_id
            final_response["session_id"] = session_id
            vibe = input_to_assess
            print(f"Updated vibe for fresh query (no previous context): '{vibe}'")
            
            # Process fresh query
            processing_result = self._process_fresh_query(vibe, current_filters)
            current_filters = processing_result["current_filters"]

        if not vibe:
            final_response["justification"] = "Original vibe description is missing, cannot proceed with targeted search."
            final_response["products"] = []
            final_response["session_id"] = context["session_id"]
            return final_response

        # Handle follow-up answer processing if this was a follow-up
        if context["user_response"] and context["last_question_text"] and context["is_follow_up"]:
            processing_result = self._process_follow_up_answer(context, vibe, current_filters)
            current_filters = processing_result["current_filters"]

        # Get refined semantic query from processing result
        refined_semantic_query = processing_result["refined_semantic_query"]
        
        # Update context and execute search
        final_response["current_filters"] = dict(current_filters)
        final_response["questions_asked_history"] = list(questions_asked_history)
        
        return self._execute_search_and_respond(vibe, current_filters, refined_semantic_query, questions_asked_history, final_response, context)

    def _process_fresh_query(self, vibe: str, current_filters: dict) -> dict:
        """Process fresh queries - parallel attribute inference and semantic query"""
        print(f"Processing fresh query with parallel LLM calls for vibe: {vibe}")
        
        is_first_meaningful_interaction = not current_filters or all(k in ['vibe_inferred'] for k in current_filters.keys())
        
        if is_first_meaningful_interaction:
            inferred_from_vibe, refined_semantic_query = self._parallel_attribute_and_semantic_calls(vibe, vibe)
            if inferred_from_vibe:
                current_filters = {**inferred_from_vibe, **current_filters} 
                current_filters["vibe_inferred"] = True
            print(f"Filters after vibe inference: {current_filters}")
        else:
            # For subsequent interactions, only need semantic query
            refined_semantic_query = self._refine_query_based_on_vibe(vibe)
        
        return {
            "vibe": vibe,
            "current_filters": current_filters,
            "refined_semantic_query": refined_semantic_query
        }

    def _process_related_query(self, context: dict, input_to_assess: str) -> dict:
        """Process related queries - combine with previous context"""
        # Combine old vibe with new input for related queries
        original_vibe = context["previous_vibe"] or context["vibe"] or ""
        if original_vibe.strip() == input_to_assess.strip():
            # If the input is identical to the original vibe, don't duplicate
            combined_vibe = original_vibe
            print(f"Input identical to original vibe, using: '{combined_vibe}'")
        else:
            combined_vibe = f"{original_vibe} {input_to_assess}".strip()
            print(f"Combined vibe: '{original_vibe}' + '{input_to_assess}' = '{combined_vibe}'")
        
        # Always do full attribute inference for consistent explicit/implicit classification  
        print(f"Running parallel attribute inference and semantic query for related query: '{input_to_assess}'")
        updated_filters, refined_semantic_query = self._parallel_attribute_and_semantic_calls(combined_vibe, combined_vibe)
        
        # Check if this is a clarification answer
        if isinstance(updated_filters, dict) and "__clarification_answer__" in updated_filters:
            # This will be handled by the caller
            pass
        
        return {
            "vibe": combined_vibe,
            "current_filters": updated_filters,
            "refined_semantic_query": refined_semantic_query
        }

    def _process_follow_up_answer(self, context: dict, vibe: str, current_filters: dict) -> dict:
        """Process follow-up answers - parallel attribute inference and semantic query"""
        print(f"📝 FOLLOW-UP ANSWER: '{context['user_response']}' to question: '{context['last_question_text']}'")
        print(f"📦 PREVIOUS FILTERS: {current_filters}")
        if vibe != context["user_response"]:  # Only log if they're actually different
            print(f"🔗 COMBINED VIBE: '{vibe}' + '{context['user_response']}'")
        
        # Always do full attribute inference for consistent explicit/implicit classification  
        previous_context = {
            "previous_vibe": vibe,
            "previous_filters": current_filters
        }
        updated_filters, refined_semantic_query = self._parallel_attribute_and_semantic_calls(context["user_response"], vibe, previous_context)
        
        # Check if this is a clarification answer
        if isinstance(updated_filters, dict) and "__clarification_answer__" in updated_filters:
            # This will be handled by the caller
            pass
        
        print(f"🔄 NEW FILTERS: {updated_filters}")
        
        return {
            "current_filters": updated_filters,
            "refined_semantic_query": refined_semantic_query
        }

    def _execute_search_and_respond(self, vibe: str, current_filters: dict, refined_semantic_query: str, 
                                   questions_asked_history: list, final_response: dict, context: dict) -> dict:
        """Execute the search and generate response with follow-up questions"""
        print(f"Proceeding to search with filters: {current_filters}")
        
        chroma_where_clause = self._build_chroma_where_clause(current_filters)
        
        print(f"DEVLOG: ChromaDB refined_semantic_query: {refined_semantic_query}")
        print(f"DEVLOG: ChromaDB where_clause: {json.dumps(chroma_where_clause, indent=2)}")
        
        top_k_target = 8
        top_k_initial_fetch = top_k_target * 3  # Increased to catch more semantic candidates and handle size filtering
        search_was_relaxed = False

        if self.collection is None or self.embedding_model is None or self.collection.count() == 0:
            final_response["justification"] = "Search service is not ready or collection is empty."
            final_response["products"] = [] 
        else:
            query_embedding_list = self.embedding_model.encode([refined_semantic_query]).tolist()
            
            try:
                chroma_query_results = self.collection.query(
                    query_embeddings=query_embedding_list,
                    n_results=top_k_initial_fetch,
                    where=chroma_where_clause if chroma_where_clause else None,
                    include=['metadatas', 'documents', 'distances']
                )
                candidate_products = []
                if chroma_query_results and chroma_query_results['ids'] and chroma_query_results['ids'][0]:
                    retrieved_ids = set()
                    for i in range(len(chroma_query_results['ids'][0])):
                        prod_id_str = chroma_query_results['ids'][0][i]
                        if prod_id_str not in retrieved_ids:
                            product_dict = chroma_query_results['metadatas'][0][i]
                            candidate_products.append(product_dict)
                            retrieved_ids.add(prod_id_str)
                
                final_products_after_py_filter = self._apply_python_filters(candidate_products, current_filters)
                
                if final_products_after_py_filter:
                    final_response["products"] = final_products_after_py_filter[:top_k_target]
                    print(f"FIRST PASS RESULTS ({len(final_response['products'])} products):")
                    for i, product in enumerate(final_response["products"]):
                        print(f"  {i+1}. [1ST PASS] {product.get('id', 'N/A')}: {product.get('name', 'N/A')}")
                else:
                    final_response["products"] = [] 

            except Exception as e:
                print(f"Error during initial ChromaDB query or processing: {e}")
                final_response["justification"] = "Error occurred during product search."
                final_response["products"] = []

            # New Relaxation Strategy: Drop All Implicit Attributes at Once
            if len(final_response["products"]) < 8:
                search_was_relaxed = self._apply_relaxation_strategy(
                    final_response, current_filters, refined_semantic_query, top_k_initial_fetch, top_k_target
                )
            
            if not final_response["products"]:
                justification_text = self._generate_justification_with_followup(vibe, [], current_filters, questions_asked_history, search_relaxed=search_was_relaxed)
                final_response["justification"] = justification_text
            else:
                final_response["justification"] = self._generate_justification_with_followup(vibe, final_response["products"], current_filters, questions_asked_history, search_relaxed=search_was_relaxed)

        # Note: Follow-up questions now integrated into justification
        # Keep questions_asked_history as is for now (can be used for tracking)
        final_response["questions_asked_history"] = list(questions_asked_history)

        # Update session storage with the final vibe used
        self._update_session_data(context["session_id"], vibe, context.get("input_to_assess", vibe))
        print(f"DEVLOG: Updated session storage - vibe: '{vibe}', input: '{context.get('input_to_assess', vibe)}'")

        # Include session ID in response
        final_response["session_id"] = context["session_id"]
        return final_response

    def _apply_relaxation_strategy(self, final_response: dict, current_filters: dict, refined_semantic_query: str,
                                  top_k_initial_fetch: int, top_k_target: int) -> bool:
        """Apply relaxation strategy to get more products"""
        print(f"Initial search yielded {len(final_response['products'])} products. Starting relaxation strategy - dropping all implicit attributes at once.")
        search_was_relaxed = False
        
        # Get attribute types from filters
        attribute_types = current_filters.get("attribute_types", {})
        
        # Define relaxation order for implicit attributes (neckline dropped first)
        implicit_relaxation_order = ['neckline', 'sleeve_length', 'length', 'fabric', 'color_or_print', 'occasion', 'fit']
        
        # Always keep explicit attributes and essential filters
        always_keep = ['category', 'size', 'price_min', 'price_max', 'budget', 'exclude_colors', 'vibe_inferred']
        
        # Start with current filters and remove all implicit attributes at once
        temp_relaxed_filters = {k: v for k, v in current_filters.items() if k != "attribute_types"}
        semantic_additions = []
        
        # Drop all implicit attributes at once
        implicit_attributes_dropped = []
        for attribute_to_drop in implicit_relaxation_order:
            # Only drop if it's an implicit attribute and not in always_keep
            if (attribute_to_drop in temp_relaxed_filters and 
                attribute_to_drop not in always_keep and
                attribute_types.get(attribute_to_drop) == "implicit"):
                
                # Move dropped attribute to semantic search
                dropped_value = temp_relaxed_filters[attribute_to_drop]
                if isinstance(dropped_value, list):
                    semantic_additions.extend([str(v) for v in dropped_value if v])
                else:
                    semantic_additions.append(str(dropped_value))
                
                # Remove from filters
                del temp_relaxed_filters[attribute_to_drop]
                implicit_attributes_dropped.append((attribute_to_drop, dropped_value))
        
        if implicit_attributes_dropped:
            search_was_relaxed = True
            print(f"🔧 RELAXATION: Dropped all implicit attributes at once: {[attr for attr, _ in implicit_attributes_dropped]} → semantic search")
            
            # Try search with all implicit attributes relaxed
            enhanced_query = f"{refined_semantic_query} {' '.join(semantic_additions)}"
            print(f"Enhanced semantic query: {enhanced_query}")
            
            enhanced_query_embedding = self.embedding_model.encode([enhanced_query])
            enhanced_query_embedding_list = [enhanced_query_embedding[0].tolist()]
            
            chroma_where_clause_relaxed = self._build_chroma_where_clause(temp_relaxed_filters)
            
            try:
                chroma_query_results_relaxed = self.collection.query(
                    query_embeddings=enhanced_query_embedding_list,
                    n_results=top_k_initial_fetch,
                    where=chroma_where_clause_relaxed if chroma_where_clause_relaxed else None,
                    include=['metadatas', 'documents', 'distances']
                )
                
                candidate_products_relaxed = []
                if chroma_query_results_relaxed and chroma_query_results_relaxed['ids'] and chroma_query_results_relaxed['ids'][0]:
                    retrieved_ids_relaxed = set()
                    for i in range(len(chroma_query_results_relaxed['ids'][0])):
                        prod_id_str_relaxed = chroma_query_results_relaxed['ids'][0][i]
                        if prod_id_str_relaxed not in retrieved_ids_relaxed:
                            product_dict_relaxed = chroma_query_results_relaxed['metadatas'][0][i]
                            candidate_products_relaxed.append(product_dict_relaxed)
                            retrieved_ids_relaxed.add(prod_id_str_relaxed)
                
                final_products_after_relaxed_py_filter = self._apply_python_filters(candidate_products_relaxed, temp_relaxed_filters)
                
                if len(final_products_after_relaxed_py_filter) >= 8:
                    print(f"RELAXATION SUCCESS: Found {len(final_products_after_relaxed_py_filter)} products after dropping all implicit attributes")
                    # Merge results
                    first_pass_products = final_response["products"]
                    first_pass_ids = {p.get("id") for p in first_pass_products}
                    
                    merged_products = first_pass_products.copy()
                    relaxed_added = 0
                    max_relaxed_to_add = 8 - len(first_pass_products)
                    
                    for product in final_products_after_relaxed_py_filter:
                        if product.get("id") not in first_pass_ids and relaxed_added < max_relaxed_to_add:
                            merged_products.append(product)
                            relaxed_added += 1
                            print(f"  {len(first_pass_products) + relaxed_added}. [RELAXED] {product.get('id', 'N/A')}: {product.get('name', 'N/A')}")
                    
                    final_response["products"] = merged_products[:top_k_target]
                else:
                    print(f"RELAXATION: Found {len(final_products_after_relaxed_py_filter)} products after dropping all implicit attributes, but need 8. Continuing to phase 2...")
                    
            except Exception as e:
                print(f"Error during relaxed search: {e}")
        
        # Phase 2: If we still have 0 products after trying all implicit attributes, try explicit too
        if len(final_response["products"]) == 0:
            search_was_relaxed = True
            print("🔧 PHASE 2: No products found, trying to drop explicit attributes")
            
            for attribute_to_drop in implicit_relaxation_order:
                # Now drop explicit attributes too (except essential ones)
                if (attribute_to_drop in temp_relaxed_filters and 
                    attribute_to_drop not in always_keep and
                    attribute_types.get(attribute_to_drop) == "explicit"):
                    
                    # Move dropped explicit attribute to semantic search
                    dropped_value = temp_relaxed_filters[attribute_to_drop]
                    if isinstance(dropped_value, list):
                        semantic_additions.extend([str(v) for v in dropped_value if v])
                    else:
                        semantic_additions.append(str(dropped_value))
                    
                    # Remove from filters
                    del temp_relaxed_filters[attribute_to_drop]
                    
                    print(f"🔧 EXPLICIT DROP: '{attribute_to_drop}' = {dropped_value} → semantic search")
                    
                    # Try search with this level of relaxation
                    enhanced_query = f"{refined_semantic_query} {' '.join(semantic_additions)}"
                    enhanced_query_embedding = self.embedding_model.encode([enhanced_query])
                    enhanced_query_embedding_list = [enhanced_query_embedding[0].tolist()]
                    
                    chroma_where_clause_relaxed = self._build_chroma_where_clause(temp_relaxed_filters)
                    
                    try:
                        chroma_query_results_relaxed = self.collection.query(
                            query_embeddings=enhanced_query_embedding_list,
                            n_results=top_k_initial_fetch,
                            where=chroma_where_clause_relaxed if chroma_where_clause_relaxed else None,
                            include=['metadatas', 'documents', 'distances']
                        )
                        
                        candidate_products_relaxed = []
                        if chroma_query_results_relaxed and chroma_query_results_relaxed['ids'] and chroma_query_results_relaxed['ids'][0]:
                            retrieved_ids_relaxed = set()
                            for i in range(len(chroma_query_results_relaxed['ids'][0])):
                                prod_id_str_relaxed = chroma_query_results_relaxed['ids'][0][i]
                                if prod_id_str_relaxed not in retrieved_ids_relaxed:
                                    product_dict_relaxed = chroma_query_results_relaxed['metadatas'][0][i]
                                    candidate_products_relaxed.append(product_dict_relaxed)
                                    retrieved_ids_relaxed.add(prod_id_str_relaxed)
                        
                        final_products_after_relaxed_py_filter = self._apply_python_filters(candidate_products_relaxed, temp_relaxed_filters)
                        
                        if len(final_products_after_relaxed_py_filter) > 0:
                            print(f"SUCCESS: Found {len(final_products_after_relaxed_py_filter)} products after dropping explicit '{attribute_to_drop}'")
                            final_response["products"] = final_products_after_relaxed_py_filter[:top_k_target]
                            break
                        else:
                            print(f"Still 0 products, continuing...")
                            
                    except Exception as e:
                        print(f"Error during explicit relaxed search for '{attribute_to_drop}': {e}")
                        continue
        
        return search_was_relaxed

    def converse(self, session_payload: dict) -> dict:
        """Main entry point for conversation processing - now clean and modular!"""
        # 1. Initialize and validate conversation context
        context = self._initialize_conversation_context(session_payload)
        
        # 2. Route to appropriate handler based on conversation type
        if context["is_follow_up"]:
            return self._handle_follow_up_response(context)
        else:
            return self._handle_initial_query(context)

product_service_instance = ProductService()
