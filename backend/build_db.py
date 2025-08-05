import pandas as pd
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
import json
import shutil

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
APPAREL_DATA_PATH = os.path.join(DATA_DIR, 'Apparels_shared.csv')
DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
VALID_ATTRIBUTES_PATH = os.path.join(DATA_DIR, 'valid_attribute_values.json')

def _load_and_process_data():
    """Loads and processes apparel data from CSV."""
    try:
        if os.path.exists(APPAREL_DATA_PATH):
            products_df = pd.read_csv(APPAREL_DATA_PATH, dtype={'id': str})
            products_df = products_df.fillna('')
            
            string_cols = ['category', 'fit', 'fabric', 'sleeve_length', 'color_or_print', 
                          'occasion', 'neckline', 'length', 'pant_type', 'name', 'description']
            for col in string_cols:
                if col in products_df.columns:
                    products_df[col] = (products_df[col].astype(str)
                                           .str.replace('\u00A0', ' ', regex=False)
                                           .str.replace('\u2000', ' ', regex=False)
                                           .str.replace('\u2001', ' ', regex=False)
                                           .str.replace('\u2002', ' ', regex=False)
                                           .str.replace('\u2003', ' ', regex=False)
                                           .str.replace('\u2004', ' ', regex=False)
                                           .str.replace('\u2005', ' ', regex=False)
                                           .str.replace('\u2006', ' ', regex=False)
                                           .str.replace('\u2007', ' ', regex=False)
                                           .str.replace('\u2008', ' ', regex=False)
                                           .str.replace('\u2009', ' ', regex=False)
                                           .str.replace('\u200A', ' ', regex=False)
                                           .str.replace('\u200B', '', regex=False)
                                           .str.replace('\u200C', '', regex=False)
                                           .str.replace('\u200D', '', regex=False)
                                           .str.replace('\uFEFF', '', regex=False)
                                           .str.strip())
            print(f"Successfully loaded {len(products_df)} products from {APPAREL_DATA_PATH}")

            product_descriptions = []
            product_ids_list = []
            description_cols = ['name', 'category', 'fit', 'fabric', 'sleeve_length',
                                'color_or_print', 'occasion', 'neckline', 'length', 'pant_type', 'description']
            for index, row in products_df.iterrows():
                desc_parts = [str(row[col]) for col in description_cols if col in row and pd.notna(row[col]) and str(row[col]).strip() != '']
                description = f"{row.get('name', '')} is a {row.get('category', '')}. "
                description += ". ".join(desc_parts[2:])
                description = description.replace("..", ".").strip()
                if description and description != ".":
                    product_descriptions.append(description)
                    product_ids_list.append(str(row['id']))
                else:
                    default_desc = f"{row.get('name', 'Product')} {row.get('category', '')}".strip()
                    product_descriptions.append(default_desc if default_desc else "Unknown Product")
                    product_ids_list.append(str(row['id']))
            
            return products_df, product_descriptions, product_ids_list
        else:
            print(f"Error: Product data file not found at {APPAREL_DATA_PATH}.")
            return None, None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None


def build_vector_store(products_df, product_descriptions, product_ids_list):
    """Builds and persists the ChromaDB vector store."""
    try:
        print("Loading SentenceTransformer model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("SentenceTransformer model loaded.")

        print(f"Generating embeddings for {len(product_descriptions)} product descriptions...")
        embeddings = embedding_model.encode(product_descriptions, show_progress_bar=True)
        embeddings_np = np.array(embeddings, dtype=np.float32).tolist()
        print("Embeddings generated.")

        print(f"Building metadata for {len(product_ids_list)} products...")
        metadatas = []
        for product_id_str in product_ids_list:
            product_data = products_df[products_df['id'] == product_id_str].iloc[0]
            meta = {
                "product_id": product_id_str,
                "id": product_id_str,
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
        print("Metadata built.")

        print(f"Initializing persistent ChromaDB client at '{DB_PATH}'...")
        if os.path.exists(DB_PATH):
            print(f"Removing existing database at '{DB_PATH}'...")
            shutil.rmtree(DB_PATH)
        
        chroma_client = chromadb.PersistentClient(path=DB_PATH)
        collection = chroma_client.get_or_create_collection(name="apparel_products")
        print("ChromaDB client and collection initialized.")

        print(f"Adding {len(product_ids_list)} items to ChromaDB collection...")
        batch_size = 5000
        for i in range(0, len(product_ids_list), batch_size):
            batch_end = i + batch_size
            print(f"Adding batch {i+1}-{min(batch_end, len(product_ids_list))}...")
            collection.add(
                embeddings=embeddings_np[i:batch_end],
                documents=product_descriptions[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=product_ids_list[i:batch_end]
            )
        
        print(f"Successfully built ChromaDB collection with {collection.count()} vectors at '{DB_PATH}'.")

    except Exception as e:
        print(f"Error building ChromaDB vector store: {e}")

def generate_valid_attributes(products_df):
    """Generates and saves valid attribute values to a JSON file."""
    try:
        print("Generating valid attribute values...")
        valid_attribute_values = {}
        attributes_to_get_values_for = [
            'category', 'fit', 'fabric', 'sleeve_length', 
            'color_or_print', 'occasion', 'neckline', 'length', 'pant_type'
        ]
        for attr in attributes_to_get_values_for:
            if attr in products_df.columns:
                unique_values = products_df[attr].dropna().astype(str).str.strip().unique()
                valid_attribute_values[attr] = sorted([val for val in unique_values if val])
        
        with open(VALID_ATTRIBUTES_PATH, 'w', encoding='utf-8') as f:
            json.dump(valid_attribute_values, f, indent=2)
        print(f"Successfully generated and saved valid attributes to {VALID_ATTRIBUTES_PATH}")
    except Exception as e:
        print(f"Error generating valid attributes: {e}")


if __name__ == "__main__":
    products_df, product_descriptions, product_ids_list = _load_and_process_data()
    if products_df is not None and product_descriptions and product_ids_list:
        build_vector_store(products_df, product_descriptions, product_ids_list)
        generate_valid_attributes(products_df)
