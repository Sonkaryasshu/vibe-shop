import pandas as pd
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
from google.genai import types
from anthropic import Anthropic
import uuid
import json
import time

# Fix HuggingFace tokenizers warning in Flask/threading environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
APPAREL_DATA_PATH = os.path.join(DATA_DIR, 'Apparels_shared.csv')
VIBE_EXAMPLES_PATH = os.path.join(DATA_DIR, 'vibe_to_attribute_examples.txt')

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
        self.products_df = None
        self.vibe_examples_text_content = ""
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.product_ids_list = []
        self.product_descriptions = []
        self.gemini_client = None
        self.gemini_pro_model = None
        self.gemini_flash_model = None
        self.anthropic_client = None
        self.claude_model = None
        self.use_claude = False  # Toggle between Gemini and Claude
        self.MAX_FOLLOW_UP_QUESTIONS = 2
        self.valid_attribute_values = {}
        
        # Backend session management
        self.session_storage = {}  # Dictionary to store session data by session_id
        self.MAX_SESSIONS = 1000  # Maximum number of sessions to keep

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
                self.use_claude = True  # Use Claude by default
                print(f"Successfully configured Anthropic API with model: {self.claude_model}. Use Claude: {self.use_claude}")
            else:
                print("Warning: ANTHROPIC_API_KEY environment variable not found. Claude LLM features will be disabled.")
                self.anthropic_client = None
        except Exception as e:
            print(f"Error configuring Anthropic API: {e}")
            self.anthropic_client = None
        
        try:
            self.chroma_client = chromadb.Client()
            self.collection = self.chroma_client.get_or_create_collection(name="apparel_products")
            print("Successfully initialized ChromaDB client and collection.")
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            self.chroma_client = None
            self.collection = None

        self._load_data()
        if self.products_df is not None and not self.products_df.empty and self.collection is not None and self.embedding_model is not None:
            self._build_vector_store()

    def _load_data(self):
        try:
            if os.path.exists(APPAREL_DATA_PATH):
                self.products_df = pd.read_csv(APPAREL_DATA_PATH)
                self.products_df = self.products_df.fillna('')
                
                # Trim whitespace and special characters from string columns
                string_cols = ['category', 'fit', 'fabric', 'sleeve_length', 'color_or_print', 
                              'occasion', 'neckline', 'length', 'pant_type', 'name', 'description']
                for col in string_cols:
                    if col in self.products_df.columns:
                        # Remove leading/trailing whitespace, NBSP, and other special whitespace chars
                        self.products_df[col] = (self.products_df[col].astype(str)
                                               .str.replace('\u00A0', ' ', regex=False)  # NBSP to regular space
                                               .str.replace('\u2000', ' ', regex=False)  # EN quad
                                               .str.replace('\u2001', ' ', regex=False)  # EM quad
                                               .str.replace('\u2002', ' ', regex=False)  # EN space
                                               .str.replace('\u2003', ' ', regex=False)  # EM space
                                               .str.replace('\u2004', ' ', regex=False)  # 3-per-EM space
                                               .str.replace('\u2005', ' ', regex=False)  # 4-per-EM space
                                               .str.replace('\u2006', ' ', regex=False)  # 6-per-EM space
                                               .str.replace('\u2007', ' ', regex=False)  # Figure space
                                               .str.replace('\u2008', ' ', regex=False)  # Punctuation space
                                               .str.replace('\u2009', ' ', regex=False)  # Thin space
                                               .str.replace('\u200A', ' ', regex=False)  # Hair space
                                               .str.replace('\u200B', '', regex=False)   # Zero-width space
                                               .str.replace('\u200C', '', regex=False)   # Zero-width non-joiner
                                               .str.replace('\u200D', '', regex=False)   # Zero-width joiner
                                               .str.replace('\uFEFF', '', regex=False)   # Zero-width no-break space (BOM)
                                               .str.strip())
                print(f"Successfully loaded {len(self.products_df)} products from {APPAREL_DATA_PATH}")

                description_cols = ['name', 'category', 'fit', 'fabric', 'sleeve_length',
                                    'color_or_print', 'occasion', 'neckline', 'length', 'pant_type', 'description']
                for index, row in self.products_df.iterrows():
                    desc_parts = [str(row[col]) for col in description_cols if col in row and pd.notna(row[col]) and str(row[col]).strip() != '']
                    description = f"{row.get('name', '')} is a {row.get('category', '')}. "
                    description += ". ".join(desc_parts[2:])
                    description = description.replace("..", ".").strip()
                    if description and description != ".":
                        self.product_descriptions.append(description)
                        self.product_ids_list.append(str(row['id']))
                    else:
                        default_desc = f"{row.get('name', 'Product')} {row.get('category', '')}".strip()
                        self.product_descriptions.append(default_desc if default_desc else "Unknown Product")
                        self.product_ids_list.append(str(row['id']))

            else:
                print(f"Warning: Product data file not found at {APPAREL_DATA_PATH}. ProductService will operate with no product data.")
                self.products_df = pd.DataFrame()

            if os.path.exists(VIBE_EXAMPLES_PATH):
                with open(VIBE_EXAMPLES_PATH, 'r', encoding='utf-8') as f:
                    self.vibe_examples_text_content = f.read()
                print(f"Successfully loaded vibe examples from {VIBE_EXAMPLES_PATH}")
            else:
                print(f"Warning: Vibe examples file not found at {VIBE_EXAMPLES_PATH}.")
                self.vibe_examples_text_content = ""

            if self.products_df is not None and not self.products_df.empty:
                attributes_to_get_values_for = [
                    'category', 'fit', 'fabric', 'sleeve_length', 
                    'color_or_print', 'occasion', 'neckline', 'length', 'pant_type'
                ]
                for attr in attributes_to_get_values_for:
                    if attr in self.products_df.columns:
                        unique_values = self.products_df[attr].dropna().astype(str).str.strip().unique()
                        self.valid_attribute_values[attr] = sorted([val for val in unique_values if val])
                print(f"Loaded valid attribute values: {json.dumps(self.valid_attribute_values, indent=2)}")


        except Exception as e:
            print(f"Error loading data: {e}")
            if self.products_df is None:
                 self.products_df = pd.DataFrame()
            if not self.vibe_examples_text_content:
                self.vibe_examples_text_content = ""
            if not hasattr(self, 'valid_attribute_values') or not self.valid_attribute_values:
                self.valid_attribute_values = {}

    def _call_llm(self, prompt: str, response_format: str = "text", thinking_budget: int = 0, use_pro_model: bool = False) -> str:
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
                print(f"Gemini Pro call took {end_time - start_time:.2f} seconds.")
                return response.candidates[0].content.parts[0].text
            except Exception as e:
                print(f"Error calling Gemini Pro API: {e}. Falling back to Claude.")
                # Fall back to Claude if Gemini Pro fails
        
        if self.use_claude and self.anthropic_client:
            try:
                start_time = time.time()
                if response_format == "json":
                    # For JSON responses with Claude
                    messages = [
                        {
                            "role": "user",
                            "content": f"{prompt}\n\nPlease respond with valid JSON only."
                        }
                    ]
                    response = self.anthropic_client.messages.create(
                        model=self.claude_model,
                        temperature=0,
                        max_tokens=5000,
                        messages=messages
                    )
                else:
                    # For text responses with Claude
                    messages = [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                    response = self.anthropic_client.messages.create(
                        model=self.claude_model,
                        temperature=0,
                        max_tokens=5000,
                        messages=messages
                    )
                end_time = time.time()
                print(f"Claude call took {end_time - start_time:.2f} seconds.")
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
                print(f"Gemini {model_name} call took {end_time - start_time:.2f} seconds.")
                return response.candidates[0].content.parts[0].text
            except Exception as e:
                print(f"Error calling Gemini API: {e}")
                return ""
        
        print("No LLM client available (neither Claude nor Gemini).")
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

        prompt = f"""
        You are a helpful assistant trying to understand if a user wants to shop for apparel.
        User's input: "{user_input}"

        Analyze this input.
        - If the input indicates interest in shopping, browsing, or learning about apparel options (e.g., "looking for a dress", "summer clothes", "what do you have?", "tell me options", "show me products", "what categories", "effortless but polished", style descriptions), then the user has shopping intent.
        - If the input is clearly unrelated to shopping for clothes (e.g., "what's the weather?", "who made you?", "how do I cook pasta?") OR is just a greeting (e.g., "hello", "hi", "hey"), then the user does not have shopping intent.
        - When in doubt, assume the user has shopping intent.
        {relatedness_section}

        Output ONLY a JSON object with these keys:
        1. "has_shopping_intent": boolean (true if shopping intent is present, false otherwise).
        2. "suggested_reply_if_no_intent": string (If `has_shopping_intent` is false, provide a polite and helpful reply to guide the user towards stating their shopping needs. If `has_shopping_intent` is true, this should be null).
        3. "is_related_query": boolean (true if related to previous context, false if completely different, null if no previous context).

        Example for "looking for a summer dress":
        {{"has_shopping_intent": true, "suggested_reply_if_no_intent": null, "is_related_query": null}}

        Example for "what do you have?":
        {{"has_shopping_intent": true, "suggested_reply_if_no_intent": null, "is_related_query": null}}

        Example for "tell me options":
        {{"has_shopping_intent": true, "suggested_reply_if_no_intent": null, "is_related_query": null}}

        Example for "effortless but polished":
        {{"has_shopping_intent": true, "suggested_reply_if_no_intent": null, "is_related_query": null}}

        Example for "what's the weather today?":
        {{"has_shopping_intent": false, "suggested_reply_if_no_intent": "I'm a shopping assistant. Are you looking for any clothing items?", "is_related_query": null}}

        Example for "hello":
        {{"has_shopping_intent": false, "suggested_reply_if_no_intent": "Hello! What kind of vibe are you looking for today?", "is_related_query": null}}

        Example with previous context - Previous: "summer dresses", New: "show full sleeves only":
        {{"has_shopping_intent": true, "suggested_reply_if_no_intent": null, "is_related_query": true}}

        Example with previous context - Previous: "summer dresses", New: "work tops that go with pants":
        {{"has_shopping_intent": true, "suggested_reply_if_no_intent": null, "is_related_query": false}}

        JSON:
        """
        try:
            response_text = self._call_llm(prompt, response_format="json", thinking_budget=0)
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

        prompt = f"""
        You are a fashion expert helping update product search filters.
        {previous_section}
        Current user input: "{vibe_description}"
        
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
        JSON:
        """
        try:
            response_text = self._call_llm(prompt, response_format="json", thinking_budget=1024)
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


    def _build_chroma_where_clause(self, filters: dict) -> dict | None:
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
        prompt_parts = [
            "You are a fashion assistant. Your task is to translate a user's desired \"vibe\" into a descriptive textual query that can be used for semantic search of apparel.",
            "Use the following examples of how vibes map to product attributes as a guide:",
            "--- VIBE EXAMPLES START ---",
            self.vibe_examples_text_content,
            "--- VIBE EXAMPLES END ---",
            f"\nUser's desired vibe: \"{vibe_description}\"",
            "\nBased on the user's vibe and the provided examples, generate a detailed textual description of product attributes that would match this vibe.",
            "Focus on characteristics like fit, fabric, color, style, occasion, patterns, and overall aesthetic.",
            "For example, if the vibe is 'classy summer wedding guest', you might describe 'elegant flowy dress, breathable fabric like silk or chiffon, possibly pastel floral print or solid light color, suitable for a formal outdoor occasion, midi or maxi length'.",
            "If the vibe is 'edgy streetwear', you might describe 'oversized graphic tee or hoodie, distressed denim or cargo pants, dark colors or bold prints, comfortable and urban style'.",
            "\nFor vague inputs like 'buy', 'shop', or 'clothes', generate a broad description covering popular versatile pieces like 'casual comfortable clothing, everyday wear pieces, versatile tops and bottoms in neutral colors, suitable for multiple occasions'.",
            "\nALWAYS generate a valid search description. Never refuse or explain why you cannot help. Output only the detailed textual description for semantic search.",
            "Detailed Description:"
        ]
        prompt = "\n".join(prompt_parts)
        try:
            response_text = self._call_llm(prompt, response_format="text", thinking_budget=256)
            if response_text:
                llm_refined_query = response_text.strip()
                print(f"LLM refined query: '{llm_refined_query}'")
                return llm_refined_query
            else:
                print("LLM response was empty. Falling back.")
        except Exception as e:
            print(f"LLM API call failed: {e}. Falling back.")
        
        print(f"Falling back to original vibe description: '{vibe_description}'")
        return vibe_description

    def _build_vector_store(self):
        if self.embedding_model is None:
            print("Error: Embedding model not loaded. Cannot build vector store.")
            return
        if self.collection is None:
            print("Error: ChromaDB collection not initialized. Cannot build vector store.")
            return
        if self.products_df is None or self.products_df.empty:
            print("Warning: Product data is empty. Cannot build vector store.")
            return
        if not self.product_descriptions or not self.product_ids_list:
            print("Warning: No product descriptions or IDs available to build vector store.")
            return
        if len(self.product_descriptions) != len(self.product_ids_list):
            print("Error: Mismatch between number of descriptions and product IDs. Cannot build vector store.")
            return

        try:
            print(f"Generating embeddings for {len(self.product_descriptions)} product descriptions...")
            embeddings = self.embedding_model.encode(self.product_descriptions, show_progress_bar=True)
            embeddings_np = np.array(embeddings, dtype=np.float32).tolist()

            print(f"Building metadata for {len(self.product_ids_list)} products...")
            metadatas = []
            for product_id_str in self.product_ids_list:
                product_data = self.products_df[self.products_df['id'] == product_id_str].iloc[0]
                meta = {
                    "product_id": product_id_str,
                    "name": str(product_data.get('name', '')),
                    "category": str(product_data.get('category', '')),
                    "price": float(product_data.get('price', 0.0)),
                    "fit": str(product_data.get('fit', '')),
                    "fabric": str(product_data.get('fabric', '')),
                    "sleeve_length": str(product_data.get('sleeve_length', '')),
                    "color_or_print": str(product_data.get('color_or_print', '')),
                    "occasion": str(product_data.get('occasion', '')),
                    "neckline": str(product_data.get('neckline', '')),
                    "length": str(product_data.get('length', '')),
                    "pant_type": str(product_data.get('pant_type', '')),
                    "available_sizes": str(product_data.get('available_sizes', '')),
                    "description": str(product_data.get('description', ''))
                }
                metadatas.append(meta)
            
            print(f"Adding {len(self.product_ids_list)} items to ChromaDB collection...")
            self.collection.add(
                embeddings=embeddings_np,
                documents=self.product_descriptions,
                metadatas=metadatas,
                ids=self.product_ids_list
            )
            print(f"Successfully built ChromaDB collection with {self.collection.count()} vectors.")

        except Exception as e:
            print(f"Error building ChromaDB vector store: {e}")

    def _generate_justification_and_followup(self, vibe_description: str, products: list, current_filters: dict, questions_asked_history: list, search_relaxed: bool = False) -> dict:
        
        filters_for_prompt = {k: v for k, v in current_filters.items() if k not in ["vibe_inferred", "attribute_types"]}

        product_details_list = []
        if products:
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
        
        product_details_string = "\n".join(product_details_list) if product_details_list else "No products found."

        filter_summary_parts = []
        for key, value in filters_for_prompt.items():
            if not value:
                continue
            if key == "price_max" and value is not None: filter_summary_parts.append(f"under ${value}")
            elif key == "price_min" and value is not None: filter_summary_parts.append(f"over ${value}")
            elif key == "size" and value: filter_summary_parts.append(f"size(s) {value if isinstance(value, str) else ', '.join(value)}")
            elif isinstance(value, list): filter_summary_parts.append(f"{key.replace('_', ' ')}: {', '.join(map(str,value))}")
            else: filter_summary_parts.append(f"{key.replace('_', ' ')}: {value}")
        filter_summary = "; ".join(filter_summary_parts)

        relaxation_instruction = ""
        if search_relaxed:
            relaxation_instruction = "\n\nIMPORTANT: The search was relaxed to find these options. Start your justification by mentioning this (e.g., 'We relaxed your search to find...' or 'After broadening criteria...')."

        no_product_justification = "No products found matching your criteria."
        if search_relaxed:
            no_product_justification = "We relaxed filters to find more options, but no products matched. Try a different style or adjust preferences."
        
        max_follow_ups_reached = len(questions_asked_history) >= self.MAX_FOLLOW_UP_QUESTIONS
        follow_up_task_instructions = f"""
        TASK 2: DETERMINE NEXT FOLLOW-UP QUESTION
        If there are still important, unclarified attributes that would significantly improve future recommendations, formulate a single, natural-sounding question to ask the user.
        Prioritize asking about: Category, Size, Budget, Fit, Occasion, or other specific style details.
        Do NOT ask about attributes already sufficiently covered in '{json.dumps(filters_for_prompt)}' or recently asked in '{questions_asked_history}'.
        If preferences are complete or max follow-ups ({self.MAX_FOLLOW_UP_QUESTIONS}) reached, no question is needed.
        """
        if max_follow_ups_reached:
            follow_up_task_instructions = "TASK 2: DO NOT ASK A FOLLOW-UP QUESTION. Maximum number of follow-ups has been reached."

        prompt = f"""You are a conversational shopping assistant. Your task is to generate a justification for recommended products and determine the next follow-up question.

        CONTEXT:
        - User's initial vibe: "{vibe_description}"
        - Current known user preferences: {filter_summary if filter_summary else "no specific additional preferences"}
        - Recommended products:
        --- RECOMMENDED PRODUCTS START ---
        {product_details_string}
        --- RECOMMENDED PRODUCTS END ---
        - Questions already asked (by their ID): {questions_asked_history}

        TASK 1: GENERATE RESPONSE TEXT
        Provide a VERY brief, concise justification (1-2 sentences maximum) explaining why these products match the user's vibe and preferences.
        If no products were found, use this text: "{no_product_justification}"
        Otherwise, be conversational and direct. Keep it under 30 words.
        {relaxation_instruction}
        
        {follow_up_task_instructions}

        If a follow-up question is generated, combine it with the justification into a single conversational paragraph. The justification should flow smoothly into the question.

        OUTPUT:
        Output your decision ONLY as a JSON object with these keys:
        - "response_text": string (The combined justification and follow-up question text. If no follow up, this is just the justification.)
        - "next_question_text": string (The question part of the text. If no question is needed, this MUST be null.)

        EXAMPLE OUTPUT (with question):
        {{
            "response_text": "These picks capture 'effortless' through relaxed fabrics and 'polished' with refined tones and tailored cuts—like the structured Mustard Muse top. To refine this further, do you have a budget or specific size in mind?",
            "next_question_text": "To refine this further, do you have a budget or specific size in mind?"
        }}
        
        EXAMPLE OUTPUT (no question needed):
        {{
            "response_text": "Based on your preferences, here are some options that match your 'edgy streetwear' vibe.",
            "next_question_text": null
        }}

        JSON:
        """

        default_response = {
            "response_text": "We found some great products for you! Their styles and features should match your vibe." if products else no_product_justification,
            "next_question_text": None
        }

        try:
            response_text = self._call_llm(prompt, response_format="json", thinking_budget=512)
            llm_response = _parse_llm_json_output(response_text)
            
            # Validate response, provide defaults if keys are missing
            if isinstance(llm_response, dict):
                # Make sure the justification for no products is correct.
                justification = llm_response.get("response_text", default_response["response_text"])
                if not products:
                    justification = no_product_justification

                return {
                    "response_text": justification,
                    "next_question_text": llm_response.get("next_question_text")
                }
            return default_response
        except Exception as e:
            print(f"Error generating justification and next question: {e}")
            return default_response

    def converse(self, session_payload: dict) -> dict:
        session_id = session_payload.get("session_id") or self._generate_session_id()
        vibe = session_payload.get("vibe_description")
        current_filters = session_payload.get("current_filters", {})
        user_response = session_payload.get("user_response")
        last_question_text = session_payload.get("last_question_text")
        questions_asked_history = session_payload.get("questions_asked_history", [])
        
        # Get previous vibe from backend session storage
        session_data = self._get_session_data(session_id)
        previous_vibe = session_data.get("previous_vibe")

        final_response = {
            "session_id": session_id,
            "follow_up_question": None,
            "question_text_for_client": None,
            "current_filters": dict(current_filters),
            "questions_asked_history": list(questions_asked_history),
            "products": None,
            "justification": None
        }

        # If user is responding to a follow-up question, check intent
        if user_response and last_question_text:
            print(f"User is responding to follow-up question: '{last_question_text}'. Checking shopping intent.")
            input_to_assess = user_response
            
            # Check if follow-up response is related to previous context or a fresh query
            print(f"📜 PREVIOUS: '{previous_vibe}'")
            print(f"💬 CURRENT: '{user_response}'")
            
            intent_assessment = self._assess_shopping_intent(user_response, previous_vibe)
            print(f"DEVLOG: Follow-up LLM assessment result: {intent_assessment}")
            has_shopping_intent = intent_assessment.get("has_shopping_intent", True)
            suggested_reply_if_no_intent = intent_assessment.get("suggested_reply_if_no_intent")
            is_related_query = intent_assessment.get("is_related_query", True)  # Default to related for follow-ups
            
            # Check if follow-up response has no shopping intent
            if not has_shopping_intent:
                print(f"Follow-up response '{user_response}' deemed to have no shopping intent.")
                final_response["justification"] = suggested_reply_if_no_intent or "How can I help you find some apparel today?"
                final_response["products"] = []
                
                # Check for context switch even for non-shopping follow-up responses
                if not is_related_query:
                    # Fresh query with no shopping intent - generate new session ID
                    session_id = self._generate_session_id()
                    print(f"Generated new session ID for fresh non-shopping follow-up: {session_id}")
                    final_response["session_id"] = session_id
                
                # Update session storage even for non-shopping follow-up inputs
                self._update_session_data(session_id, vibe or input_to_assess, input_to_assess)
                print(f"DEVLOG: Updated session storage for non-shopping follow-up - vibe: '{vibe or input_to_assess}', input: '{input_to_assess}'")
                
                return final_response
        else:
            # Only assess intent for initial interactions
            input_to_assess = ""
            if user_response:
                input_to_assess = user_response
            elif vibe:
                input_to_assess = vibe
            
            if not input_to_assess:
                final_response["justification"] = "Hello! How can I help you find some apparel today?"
                final_response["products"] = []
                final_response["session_id"] = session_id
                return final_response

            # Get previous vibe for relatedness assessment
            print(f"📜 PREVIOUS: '{previous_vibe}'")
            print(f"💬 CURRENT: '{input_to_assess}'")
            intent_assessment = self._assess_shopping_intent(input_to_assess, previous_vibe)
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
                
                # Update session storage even for non-shopping inputs
                self._update_session_data(session_id, vibe or input_to_assess, input_to_assess)
                print(f"DEVLOG: Updated session storage for non-shopping input - vibe: '{vibe or input_to_assess}', input: '{input_to_assess}'")
                
                return final_response
        
        print(f"Input '{input_to_assess}' has shopping intent. Proceeding with product logic.")

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
                # Update vibe to the new query (use the current input as the new vibe)
                vibe = input_to_assess
                print(f"Updated vibe for fresh query: '{vibe}'")
            else:
                print("Related query detected - retaining context and combining vibe.")
                # Combine old vibe with new input for related queries - use previous_vibe from session
                # Avoid duplicating the same vibe description
                original_vibe = previous_vibe or vibe or ""
                if original_vibe.strip() == input_to_assess.strip():
                    # If the input is identical to the original vibe, don't duplicate
                    combined_vibe = original_vibe
                    print(f"Input identical to original vibe, using: '{combined_vibe}'")
                else:
                    combined_vibe = f"{original_vibe} \n\n\n {input_to_assess}".strip()
                    print(f"Combined vibe: '{original_vibe}' + '{input_to_assess}' = '{combined_vibe}'")
                vibe = combined_vibe
                
                # Always do full attribute inference for consistent explicit/implicit classification
                print(f"Doing full attribute inference for related query: '{input_to_assess}'")
                updated_filters = self._infer_attributes_from_vibe(combined_vibe)
                
                # Check if this is a clarification answer
                if isinstance(updated_filters, dict) and "__clarification_answer__" in updated_filters:
                    final_response["justification"] = updated_filters["__clarification_answer__"]
                    final_response["products"] = []
                    final_response["session_id"] = session_id
                    return final_response
                
                current_filters = updated_filters
                print(f"DEVLOG: Filters after extracting from related query: {current_filters}")
        else:
            # When relatedness cannot be determined (no previous context), treat as fresh query
            print("No previous context available - treating as fresh query.")
            current_filters = {}
            questions_asked_history = []
            # Generate new session ID for fresh query (no previous context)
            session_id = self._generate_session_id()
            print(f"Generated new session ID for fresh query (no previous context): {session_id}")
            vibe = input_to_assess
            print(f"Updated vibe for fresh query (no previous context): '{vibe}'")

        if not vibe:
            final_response["justification"] = "Original vibe description is missing, cannot proceed with targeted search."
            final_response["products"] = []
            final_response["session_id"] = session_id
            return final_response

        if user_response and last_question_text:
            print(f"📝 FOLLOW-UP ANSWER: '{user_response}' to question: '{last_question_text}'")
            print(f"📦 PREVIOUS FILTERS: {current_filters}")
            if vibe != user_response:  # Only log if they're actually different
                print(f"🔗 COMBINED VIBE: '{vibe}' + '{user_response}'")
            
            # Always do full attribute inference for consistent explicit/implicit classification  
            previous_context = {
                "previous_vibe": vibe,
                "previous_filters": current_filters
            }
            updated_filters = self._infer_attributes_from_vibe(user_response, previous_context)
            
            # Check if this is a clarification answer
            if isinstance(updated_filters, dict) and "__clarification_answer__" in updated_filters:
                final_response["justification"] = updated_filters["__clarification_answer__"]
                final_response["products"] = []
                final_response["session_id"] = session_id
                return final_response
            
            print(f"🔄 NEW FILTERS: {updated_filters}")
            current_filters = updated_filters

        is_first_meaningful_interaction = not questions_asked_history and \
                                         (not current_filters or all(k in ['vibe_inferred'] for k in current_filters.keys()))

        if is_first_meaningful_interaction:
            print(f"First interaction or minimal filters. Inferring from vibe: {vibe}")
            inferred_from_vibe = self._infer_attributes_from_vibe(vibe)
            if inferred_from_vibe:
                current_filters = {**inferred_from_vibe, **current_filters} 
                current_filters["vibe_inferred"] = True
            print(f"Filters after vibe inference: {current_filters}")
        
        final_response["current_filters"] = dict(current_filters)
        final_response["questions_asked_history"] = list(questions_asked_history)

        print(f"Proceeding to search with filters: {current_filters}")
        refined_semantic_query = self._refine_query_based_on_vibe(vibe)
        chroma_where_clause = self._build_chroma_where_clause(current_filters)
        
        print(f"DEVLOG: ChromaDB refined_semantic_query: {refined_semantic_query}")
        print(f"DEVLOG: ChromaDB where_clause: {json.dumps(chroma_where_clause, indent=2)}")
        
        top_k_target = 8
        top_k_initial_fetch = top_k_target * 4 

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
                            product_series_df = self.products_df[self.products_df['id'] == prod_id_str]
                            if not product_series_df.empty:
                                product_dict = product_series_df.iloc[0].to_dict()
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

            # New Relaxation Strategy: Ordered Implicit Attribute Removal
            if len(final_response["products"]) < top_k_target:
                print(f"Initial search yielded {len(final_response['products'])} products. Starting ordered relaxation strategy.")
                
                # Get attribute types from filters
                attribute_types = current_filters.get("attribute_types", {})
                
                # Define relaxation order for implicit attributes (neckline dropped first)
                implicit_relaxation_order = ['neckline', 'sleeve_length', 'length', 'fabric', 'color_or_print', 'occasion', 'fit']
                
                # Always keep explicit attributes and essential filters
                always_keep = ['category', 'size', 'price_min', 'price_max', 'budget', 'exclude_colors', 'vibe_inferred']
                
                # Start with current filters and progressively remove implicit attributes
                temp_relaxed_filters = {k: v for k, v in current_filters.items() if k != "attribute_types"}
                semantic_additions = []
                
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
                        
                        print(f"🔧 RELAXATION: Dropped implicit '{attribute_to_drop}' = {dropped_value} → semantic search")
                        
                        # Try search with this level of relaxation
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
                                        product_series_df_relaxed = self.products_df[self.products_df['id'] == prod_id_str_relaxed]
                                        if not product_series_df_relaxed.empty:
                                            product_dict_relaxed = product_series_df_relaxed.iloc[0].to_dict()
                                            candidate_products_relaxed.append(product_dict_relaxed)
                                            retrieved_ids_relaxed.add(prod_id_str_relaxed)
                            
                            final_products_after_relaxed_py_filter = self._apply_python_filters(candidate_products_relaxed, temp_relaxed_filters)
                            
                            if len(final_products_after_relaxed_py_filter) >= 3:
                                print(f"RELAXATION SUCCESS: Found {len(final_products_after_relaxed_py_filter)} products after dropping '{attribute_to_drop}'")
                                # Merge results and break
                                first_pass_products = final_response["products"]
                                first_pass_ids = {p.get("id") for p in first_pass_products}
                                
                                merged_products = first_pass_products.copy()
                                relaxed_added = 0
                                max_relaxed_to_add = top_k_target - len(first_pass_products)
                                
                                for product in final_products_after_relaxed_py_filter:
                                    if product.get("id") not in first_pass_ids and relaxed_added < max_relaxed_to_add:
                                        merged_products.append(product)
                                        relaxed_added += 1
                                        print(f"  {len(first_pass_products) + relaxed_added}. [RELAXED] {product.get('id', 'N/A')}: {product.get('name', 'N/A')}")
                                
                                final_response["products"] = merged_products[:top_k_target]
                                break
                            else:
                                print(f"RELAXATION: Still only {len(final_products_after_relaxed_py_filter)} products, continuing...")
                                
                        except Exception as e:
                            print(f"Error during relaxed search for '{attribute_to_drop}': {e}")
                            continue
                
                # If we still have 0 products after trying all implicit attributes, try explicit too
                if len(final_response["products"]) == 0:
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
                            search_was_relaxed = True
                            
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
                                            product_series_df_relaxed = self.products_df[self.products_df['id'] == prod_id_str_relaxed]
                                            if not product_series_df_relaxed.empty:
                                                product_dict_relaxed = product_series_df_relaxed.iloc[0].to_dict()
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
                
                # Mark as relaxed if we tried any relaxation
                if len(final_response["products"]) == 0:
                    search_was_relaxed = True
            
            # Generate justification and determine next follow-up in a single LLM call
            justification_and_followup = self._generate_justification_and_followup(
                vibe,
                final_response["products"] or [],
                current_filters,
                questions_asked_history,
                search_relaxed=search_was_relaxed
            )

            justification_text = justification_and_followup.get("response_text")
            next_q_text = justification_and_followup.get("next_question_text")
            
            final_response["justification"] = justification_text
            
            if next_q_text:
                # Still populate follow_up_question fields for conversation state tracking
                final_response["follow_up_question"] = next_q_text
                final_response["question_text_for_client"] = next_q_text
                final_response["questions_asked_history"] = questions_asked_history + [next_q_text]
                print(f"Suggesting follow-up: '{next_q_text}' alongside results.")
            else:
                print("No further follow-up question suggested or limit reached.")
                final_response["questions_asked_history"] = list(questions_asked_history)

        # Update session storage with the final vibe used
        self._update_session_data(session_id, vibe, input_to_assess)
        print(f"DEVLOG: Updated session storage - vibe: '{vibe}', input: '{input_to_assess}'")

        # Include session ID in response
        final_response["session_id"] = session_id

        return final_response

product_service_instance = ProductService()
