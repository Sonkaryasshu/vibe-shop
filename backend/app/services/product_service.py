import pandas as pd
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
from google.genai import types
import uuid
import json
import time

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
        self.MAX_FOLLOW_UP_QUESTIONS = 2
        self.valid_attribute_values = {}
        
        # Backend session management
        self.session_storage = {}  # Dictionary to store session data by session_id

        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Successfully loaded SentenceTransformer model.")
        except Exception as e:
            print(f"Error loading SentenceTransformer model: {e}")
            self.embedding_model = None
        
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

    def _get_session_data(self, session_id: str) -> dict:
        """Get session data for a given session ID"""
        if session_id not in self.session_storage:
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
        # Keep only last 10 interactions to avoid memory bloat
        if len(session_data["conversation_history"]) > 10:
            session_data["conversation_history"] = session_data["conversation_history"][-10:]
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        return f"session_{uuid.uuid4().hex[:12]}"

    def _assess_shopping_intent(self, user_input: str, previous_vibe: str = None) -> dict:
        if not self.gemini_client:
            print("Gemini client not available for assessing shopping intent. Defaulting to has_shopping_intent: True.")
            return {"has_shopping_intent": True, "suggested_reply_if_no_intent": None, "is_related_query": True}
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
            start_time = time.time()
            response = self.gemini_client.models.generate_content(
                model=self.gemini_flash_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=0  # No thinking needed for simple intent detection
                    ),
                    response_mime_type="application/json"
                )
            )
            end_time = time.time()
            print(f"Gemini Flash call to _assess_shopping_intent took {end_time - start_time:.2f} seconds.")
            
            assessment_result = _parse_llm_json_output(response.candidates[0].content.parts[0].text)
            if isinstance(assessment_result, dict) and "has_shopping_intent" in assessment_result:
                return {
                    "has_shopping_intent": assessment_result.get("has_shopping_intent", False),
                    "suggested_reply_if_no_intent": assessment_result.get("suggested_reply_if_no_intent"),
                    "is_related_query": assessment_result.get("is_related_query", None)
                }
            else:
                print(f"Failed to parse valid shopping intent assessment from LLM: {response.text}. Defaulting to has_shopping_intent: True.")
                return {"has_shopping_intent": True, "suggested_reply_if_no_intent": None, "is_related_query": None}
        except Exception as e:
            print(f"Error during _assess_shopping_intent with Gemini: {e}. Defaulting to has_shopping_intent: True.")
            return {"has_shopping_intent": True, "suggested_reply_if_no_intent": None, "is_related_query": None}

    def _infer_attributes_from_vibe(self, vibe_description: str) -> dict:
        if not self.gemini_client:
            print("Gemini client not available for inferring attributes from vibe.")
            return {}

        prompt = f"""
        You are a fashion expert. Given the user's vibe: "{vibe_description}"
        And the following examples of vibe-to-attribute mappings:
        --- VIBE EXAMPLES START ---
        {self.vibe_examples_text_content}
        --- VIBE EXAMPLES END ---

        Here are the ONLY VALID values for certain filterable attributes. When inferring these attributes, you MUST choose from these lists if you decide to include the attribute.
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
        Be conservative. If an attribute is not strongly implied or a valid value cannot be found, do not include it.
        Output your answer ONLY as a JSON object. For example:
        {{"category": ["dress"], "fabric": ["linen", "cotton"], "occasion": "summer brunch"}}
        Another example, if vibe is "plus sized summer party under $75": {{"occasion": "party", "size": ["XL", "XXL"], "fabric": ["cotton", "linen"], "price_max": 75}}
        Another example, if vibe is "work pants between $60 and $120": {{"category": ["pants"], "occasion": "work", "price_min": 60, "price_max": 120}}
        Another example, if vibe is "sleeveless dresses": {{"category": ["dress"], "sleeve_length": "Sleeveless"}}
        Another example, if vibe is "work blouses, no black or blue": {{"category": ["top"], "occasion": "work", "exclude_colors": ["black", "blue"]}}
        Another example, if vibe is "tops and dresses, only sleeveless": {{"category": ["top", "dress"], "sleeve_length": "Sleeveless"}}
        If no attributes can be confidently inferred, output an empty JSON object {{}}.
        JSON:
        """
        try:
            start_time = time.time()
            response = self.gemini_client.models.generate_content(
                model=self.gemini_flash_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=512  # Flash with moderate thinking for attribute inference
                    ),
                    response_mime_type="application/json"
                )
            )
            end_time = time.time()
            print(f"Gemini Flash call to _infer_attributes_from_vibe took {end_time - start_time:.2f} seconds.")
            inferred_attributes = _parse_llm_json_output(response.candidates[0].content.parts[0].text)

            if inferred_attributes and self.valid_attribute_values:
                validated_attributes = {}
                for key, value in inferred_attributes.items():
                    if key in self.valid_attribute_values:
                        valid_options_for_key = self.valid_attribute_values[key]
                        if isinstance(value, list):
                            cleaned_values = [v for v in value if v in valid_options_for_key]
                            if cleaned_values:
                                validated_attributes[key] = cleaned_values
                        elif isinstance(value, str):
                            if value in valid_options_for_key:
                                validated_attributes[key] = value
                    else:
                        validated_attributes[key] = value
                
                print(f"Original inferred attributes: {inferred_attributes}")
                print(f"Validated attributes: {validated_attributes}")
                return validated_attributes
            else:
                return inferred_attributes

        except Exception as e:
            print(f"Error inferring attributes from vibe with Gemini: {e}")
            return {}

    def _parse_user_answer_and_update_filters(self, last_question_text: str, user_answer: str, current_filters: dict) -> dict:
        if not self.gemini_client:
            print("Gemini client not available for parsing user answer.")
            return current_filters

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
            start_time = time.time()
            response = self.gemini_client.models.generate_content(
                model=self.gemini_flash_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=0  # No thinking needed for simple parsing
                    ),
                    response_mime_type="application/json"
                )
            )
            end_time = time.time()
            print(f"Gemini Flash call to _parse_user_answer_and_update_filters took {end_time - start_time:.2f} seconds.")
            
            llm_suggested_changes = _parse_llm_json_output(response.candidates[0].content.parts[0].text)
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
            print(f"Error parsing user answer with Gemini: {e}")
            return current_filters

    def _determine_next_follow_up(self, vibe_description: str, current_filters: dict, questions_asked_history: list) -> tuple[str | None, str | None, str | None]:
        if not self.gemini_client:
            print("Gemini client not available for determining follow-up.")
            return None, None, None
        
        if len(questions_asked_history) >= self.MAX_FOLLOW_UP_QUESTIONS:
            print(f"Max follow-up questions ({self.MAX_FOLLOW_UP_QUESTIONS}) reached or exceeded. Not asking another.")
            return None, None, None

        filters_for_prompt = {k: v for k, v in current_filters.items() if k != "vibe_inferred"}

        prompt = f"""
        You are a conversational shopping assistant. Product recommendations may have just been shown or are about to be shown based on current information.
        User's initial vibe: "{vibe_description}"
        Current known user preferences: {json.dumps(filters_for_prompt)}
        Questions already asked (by their ID): {questions_asked_history}
        Number of follow-up questions asked so far: {len(questions_asked_history)}. Max {self.MAX_FOLLOW_UP_QUESTIONS} follow-ups in total.

        Your task: If there are still important, unclarified attributes that would significantly improve future recommendations, formulate a single, natural-sounding question to ask the user.
        Prioritize asking about the following key aspects if they are missing or unclear from "Current known user preferences":
        - Category (e.g., dress, top, pants)
        - Size
        - Budget (price range, e.g., under $100, $50-$150)
        - Fit (e.g., relaxed, tailored)
        - Occasion (e.g., work, casual, party)
        - Other specific style details (e.g., Sleeve Length, Garment Length, Color/Print, Fabric type).
        This question will be shown alongside the current product recommendations to help refine the next search.

        IMPORTANT:
        1.  Examine "Current known user preferences" and "Questions already asked" VERY CAREFULLY.
            Do NOT ask about attributes already sufficiently covered or recently asked.
        2.  If current preferences seem reasonably complete for good recommendations OR if all {self.MAX_FOLLOW_UP_QUESTIONS} follow-up questions have been asked,
            indicate that no further question is needed by returning nulls.
        3.  The question should ideally target 1-2 key missing pieces of information.
            Example: If budget and size are still vague: "To refine this further, do you have a budget or specific size in mind?"
            Example: If specific style details are missing: "Any other preferences, perhaps for sleeve length or fit, to narrow it down more?"

        Output your decision ONLY as a JSON object with three keys: "next_question_text", "next_question_id", "attribute_key".
        - "next_question_text": The question to ask. If no question is needed, this should be null.
        - "next_question_id": A concise ID for the question (e.g., "ask_size_budget", "ask_style_details"). If no question, null.
        - "attribute_key": The primary filter key(s) this question relates to (e.g., "size,price_max", "sleeve_length,fit", "category"). If no question, null.
        
        Example JSON if asking a question: {{"next_question_text": "Great. Any must-haves like size or a budget to keep in mind?", "next_question_id": "ask_size_budget", "attribute_key": "size,price_max"}}
        Example JSON if no question needed: {{"next_question_text": null, "next_question_id": null, "attribute_key": null}}
        JSON:
        """
        try:
            start_time = time.time()
            response = self.gemini_client.models.generate_content(
                model=self.gemini_flash_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=0  # No thinking needed for simple follow-up logic
                    ),
                    response_mime_type="application/json"
                )
            )
            end_time = time.time()
            print(f"Gemini Flash call to _determine_next_follow_up took {end_time - start_time:.2f} seconds.")
            decision = _parse_llm_json_output(response.candidates[0].content.parts[0].text)
            
            if decision.get("next_question_text") is None:
                return None, None, None
            
            if decision.get("next_question_id") and decision.get("attribute_key"):
                 return decision.get("next_question_text"), decision.get("next_question_id"), decision.get("attribute_key")
            else:
                print("LLM suggested a question but was missing id or attribute_key. Treating as no question.")
                return None, None, None

        except Exception as e:
            print(f"Error determining next follow-up with Gemini: {e}")
            return None, None, None

    def _build_chroma_where_clause(self, filters: dict) -> dict | None:
        where_conditions = []
        processed_keys = set()

        if "price_min" in filters and "price_max" in filters and filters["price_min"] is not None and filters["price_max"] is not None:
            where_conditions.append({"$and": [{"price": {"$gte": float(filters["price_min"])}}, {"price": {"$lte": float(filters["price_max"])}}]})
        elif "price_min" in filters and filters["price_min"] is not None:
            where_conditions.append({"price": {"$gte": float(filters["price_min"])}})
        elif "price_max" in filters and filters["price_max"] is not None:
            where_conditions.append({"price": {"$lte": float(filters["price_max"])}})
        processed_keys.update(["price_min", "price_max", "budget", "vibe_inferred", "exclude_colors"])


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
        if self.gemini_client:
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
                "\nOutput only the detailed textual description for semantic search. Do not add any conversational fluff.",
                "Detailed Description:"
            ]
            prompt = "\n".join(prompt_parts)
            try:
                start_time = time.time()
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_flash_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=256  # Lighter thinking for query refinement
                        )
                    )
                )
                end_time = time.time()
                print(f"Gemini Flash call to _refine_query_based_on_vibe took {end_time - start_time:.2f} seconds.")
                if response.candidates[0].content.parts[0].text:
                    llm_refined_query = response.candidates[0].content.parts[0].text.strip()
                    print(f"Gemini refined query: '{llm_refined_query}'")
                    return llm_refined_query
                else:
                    print("Gemini response was empty. Falling back.")
            except Exception as e:
                print(f"Gemini API call failed: {e}. Falling back.")
        
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

    def _generate_justification(self, vibe_description: str, products: list, current_filters: dict, search_relaxed: bool = False) -> str:
        if not self.gemini_client:
            return "Could not generate justification as the language model is not available."

        if not products:
            if search_relaxed:
                return "Even after broadening the search, no products matched your criteria. Perhaps try a different vibe or adjust your preferences?"
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

        prompt_prefix = ""
        if search_relaxed:
            prompt_prefix = "To find some options for you, I broadened the search slightly from your initial criteria. "

        prompt = f"""{prompt_prefix}The user expressed a desire for products matching the vibe: "{vibe_description}".
Additionally, they specified the following preferences: {filter_summary if filter_summary else "no specific additional preferences"}.

Based on this, we have recommended the following products:
--- RECOMMENDED PRODUCTS START ---
{product_details_string}
--- RECOMMENDED PRODUCTS END ---

Please provide a VERY brief, concise justification (1-2 sentences maximum) explaining why these products match their vibe.
Focus on the key attributes that align with the vibe. Be conversational and direct.

Example format: "These picks capture 'effortless' through relaxed fabrics and 'polished' with refined tones and tailored cuts—like the structured Mustard Muse top."

IMPORTANT: Keep it under 30 words. Be specific about how the products match the vibe, not generic descriptions.

Justification:
"""
        try:
            start_time = time.time()
            response = self.gemini_client.models.generate_content(
                model=self.gemini_flash_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=512  # Flash with moderate thinking for justification
                    )
                )
            )
            end_time = time.time()
            print(f"Gemini Flash call to _generate_justification took {end_time - start_time:.2f} seconds.")
            return response.candidates[0].content.parts[0].text.strip()
        except Exception as e:
            print(f"Error generating justification with Gemini: {e}")
            return "We found some great products for you! Their styles and features should match your vibe."

    def converse(self, session_payload: dict) -> dict:
        session_id = session_payload.get("session_id", "default_session")
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
            "question_id": None,
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
            print(f"DEVLOG: Follow-up - previous_vibe: '{previous_vibe}'")
            print(f"DEVLOG: Follow-up - user_response: '{user_response}'")
            
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
                return final_response

            # Get previous vibe for relatedness assessment
            print(f"DEVLOG: previous_vibe from session: '{previous_vibe}'")
            print(f"DEVLOG: input_to_assess: '{input_to_assess}'")
            intent_assessment = self._assess_shopping_intent(input_to_assess, previous_vibe)
            print(f"DEVLOG: LLM intent assessment result: {intent_assessment}")
            has_shopping_intent = intent_assessment.get("has_shopping_intent", True)
            suggested_reply_if_no_intent = intent_assessment.get("suggested_reply_if_no_intent")
            is_related_query = intent_assessment.get("is_related_query", None)

            if not has_shopping_intent:
                print(f"Input '{input_to_assess}' deemed to have no shopping intent.")
                final_response["justification"] = suggested_reply_if_no_intent or "How can I help you find some apparel today?"
                final_response["products"] = []
                return final_response
        
        print(f"Input '{input_to_assess}' has shopping intent. Proceeding with product logic.")

        # Handle query context management - related vs fresh queries
        if is_related_query is not None:
            print(f"Query relatedness assessment: {'related' if is_related_query else 'fresh'}")
            if not is_related_query:
                # Fresh query - reset filters and questions history
                print("Fresh query detected - resetting context.")
                current_filters = {}
                questions_asked_history = []
                # Update vibe to the new query (use the current input as the new vibe)
                vibe = input_to_assess
                print(f"Updated vibe for fresh query: '{vibe}'")
            else:
                print("Related query detected - retaining context and combining vibe.")
                # Combine old vibe with new input for related queries
                original_vibe = vibe
                combined_vibe = f"{original_vibe} {input_to_assess}"
                vibe = combined_vibe
                print(f"Combined vibe: '{original_vibe}' + '{input_to_assess}' = '{vibe}'")
                
                # Extract any attribute updates from related queries (like size changes)
                print(f"Extracting attributes from related query: '{input_to_assess}'")
                updated_filters = self._parse_user_answer_and_update_filters("", input_to_assess, current_filters)
                
                # Check if this is a clarification answer
                if isinstance(updated_filters, dict) and "__clarification_answer__" in updated_filters:
                    final_response["justification"] = updated_filters["__clarification_answer__"]
                    final_response["products"] = []
                    return final_response
                
                current_filters = updated_filters
                print(f"DEVLOG: Filters after extracting from related query: {current_filters}")
        else:
            # When relatedness cannot be determined (no previous context), treat as fresh query
            print("No previous context available - treating as fresh query.")
            current_filters = {}
            questions_asked_history = []
            vibe = input_to_assess
            print(f"Updated vibe for fresh query (no previous context): '{vibe}'")

        if not vibe:
            final_response["justification"] = "Original vibe description is missing, cannot proceed with targeted search."
            final_response["products"] = []
            return final_response

        if user_response and last_question_text:
            print(f"Parsing user response: '{user_response}' to question: '{last_question_text}' with current filters: {current_filters}")
            updated_filters = self._parse_user_answer_and_update_filters(last_question_text, user_response, current_filters)
            
            # Check if this is a clarification answer
            if isinstance(updated_filters, dict) and "__clarification_answer__" in updated_filters:
                final_response["justification"] = updated_filters["__clarification_answer__"]
                final_response["products"] = []
                return final_response
            
            current_filters = updated_filters
            print(f"Filters after parsing answer: {current_filters}")

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
        top_k_initial_fetch = top_k_target 
        if "size" in current_filters and current_filters["size"]:
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
                else:
                    final_response["products"] = [] 

            except Exception as e:
                print(f"Error during initial ChromaDB query or processing: {e}")
                final_response["justification"] = "Error occurred during product search."
                final_response["products"] = []

            # Relaxation Logic if initial search failed
            if not final_response["products"]:
                print("Initial search yielded no products. Attempting to relax filters.")
                
                # Keep only category and size filters, move others to semantic search
                filters_to_keep = ['category', 'size', 'price_min', 'price_max', 'budget']
                filters_to_semanticize = ['color_or_print', 'occasion', 'fabric', 'fit', 
                                         'sleeve_length', 'length', 'neckline', 'pant_type']
                
                temp_relaxed_filters = {}
                semantic_additions = []
                
                # Keep only essential filters for ChromaDB
                for key in filters_to_keep:
                    if key in current_filters and current_filters[key] is not None:
                        temp_relaxed_filters[key] = current_filters[key]
                
                # Collect removed filter values for semantic search
                for key in filters_to_semanticize:
                    if key in current_filters and current_filters[key] is not None:
                        value = current_filters[key]
                        if isinstance(value, list):
                            semantic_additions.extend([str(v) for v in value if v])
                        else:
                            semantic_additions.append(str(value))
                
                if semantic_additions or temp_relaxed_filters:
                    # Enhance the semantic query with filter values if available
                    if semantic_additions:
                        enhanced_query = f"{refined_semantic_query} {' '.join(semantic_additions)}"
                        print(f"Enhanced semantic query with filters: {enhanced_query}")
                        
                        # Re-generate embedding for enhanced query
                        enhanced_query_embedding = self.embedding_model.encode([enhanced_query])
                        enhanced_query_embedding_list = [enhanced_query_embedding[0].tolist()]
                        
                        print(f"Relaxed to keep only: {list(temp_relaxed_filters.keys())}. Added to semantic search: {semantic_additions}")
                    else:
                        # No semantic additions, use original query
                        enhanced_query_embedding_list = query_embedding_list
                        print(f"Relaxed to keep only: {list(temp_relaxed_filters.keys())}. No additional semantic terms.")
                    
                    search_was_relaxed = True

                    chroma_where_clause_relaxed = self._build_chroma_where_clause(temp_relaxed_filters)
                    print(f"DEVLOG: Relaxed ChromaDB where_clause: {json.dumps(chroma_where_clause_relaxed, indent=2)}")

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
                        
                        if final_products_after_relaxed_py_filter:
                            final_response["products"] = final_products_after_relaxed_py_filter[:top_k_target]
                            print(f"Found {len(final_response['products'])} products after relaxing filters.")
                        else:
                            final_response["products"] = []
                            print("Still no products found even after relaxing filters.")
                    except Exception as e:
                        print(f"Error during relaxed ChromaDB query or processing: {e}")
                        final_response["products"] = []
                else:
                    print("No relaxable filters were present in current_filters. Cannot relax further.")
            
            if not final_response["products"]:
                justification_text = self._generate_justification(vibe, [], current_filters, search_relaxed=search_was_relaxed)
                final_response["justification"] = justification_text
            else:
                final_response["justification"] = self._generate_justification(vibe, final_response["products"], current_filters, search_relaxed=search_was_relaxed)

        next_q_text, next_q_id, next_q_attr_key = self._determine_next_follow_up(vibe, current_filters, questions_asked_history)
        
        if next_q_text and next_q_id:
            final_response["follow_up_question"] = next_q_text
            final_response["question_id"] = next_q_id
            final_response["question_text_for_client"] = next_q_text
            final_response["questions_asked_history"] = questions_asked_history + [next_q_id]
            print(f"Suggesting follow-up: '{next_q_text}' (ID: {next_q_id}) alongside results.")
        else:
            print("No further follow-up question suggested or limit reached.")
            final_response["questions_asked_history"] = list(questions_asked_history)

        # Update session storage with the final vibe used
        self._update_session_data(session_id, vibe, input_to_assess)
        print(f"DEVLOG: Updated session storage - vibe: '{vibe}', input: '{input_to_assess}'")

        return final_response

product_service_instance = ProductService()
