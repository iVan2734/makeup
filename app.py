from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import json
import os
import csv
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import requests
from urllib.parse import urlparse
import dupes

app = Flask(__name__)
CORS(app)

# Dataset will be loaded from file
MAKEUP_DATASET = []

def load_dataset(file_path: str = None):
    """Load dataset from JSON or CSV file"""
    global MAKEUP_DATASET
    
    # Default dataset file paths
    json_path = os.path.join(os.path.dirname(__file__), 'dataset.json')
    csv_path = os.path.join(os.path.dirname(__file__), 'dataset.csv')
    
    # Use provided path or try default paths
    if file_path:
        path = file_path
    elif os.path.exists(json_path):
        path = json_path
    elif os.path.exists(csv_path):
        path = csv_path
    else:
        # Fallback to empty dataset
        MAKEUP_DATASET = []
        return
    
    try:
        if path.endswith('.json'):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Handle both list and dict formats
                if isinstance(data, list):
                    MAKEUP_DATASET = data
                elif isinstance(data, dict) and 'items' in data:
                    MAKEUP_DATASET = data['items']
                else:
                    MAKEUP_DATASET = []
        elif path.endswith('.csv'):
            MAKEUP_DATASET = []
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader, start=1):
                    # Ensure ID exists
                    if 'id' not in row or not row['id']:
                        row['id'] = idx
                    else:
                        row['id'] = int(row['id']) if row['id'].isdigit() else idx
                    MAKEUP_DATASET.append(row)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        MAKEUP_DATASET = []

# Load dataset on startup - try makeup_data.json first
if os.path.exists(os.path.join(os.path.dirname(__file__), 'makeup_data.json')):
    load_dataset(os.path.join(os.path.dirname(__file__), 'makeup_data.json'))
else:
    load_dataset()

def find_duplicates_basic(dataset):
    """Find duplicate makeup items based on exact matching of key fields"""
    seen = {}
    duplicates = []
    
    for item in dataset:
        # Create a key from the identifying fields (normalize strings)
        # Handle both old format (type, color) and new format (product_type, product_colors)
        name = str(item.get("name", "")).strip().lower()
        brand = str(item.get("brand", "")).strip().lower()
        
        # Try product_type first (new format), then type (old format)
        item_type = str(item.get("product_type", item.get("type", ""))).strip().lower()
        
        # For color, use product_colors if available, otherwise use color field
        # If product_colors exists, we consider items with same name/brand/type as duplicates
        # regardless of specific colors (same product, different shades)
        has_product_colors = item.get("product_colors") and len(item.get("product_colors", [])) > 0
        
        if has_product_colors:
            # For items with product_colors, match on name + brand + product_type only
            # (same product can have multiple color variants)
            key = (name, brand, item_type)
        else:
            # For items without product_colors, also match on color field if available
            color = str(item.get("color", "")).strip().lower()
            key = (name, brand, item_type, color)
        
        if key in seen:
            # Found a duplicate
            if key not in [d["key"] for d in duplicates]:
                # First time seeing this duplicate, add the original
                duplicates.append({
                    "key": key,
                    "original": seen[key],
                    "duplicates": [item],
                    "confidence": "exact"
                })
            else:
                # Add to existing duplicate group
                for dup_group in duplicates:
                    if dup_group["key"] == key:
                        dup_group["duplicates"].append(item)
                        break
        else:
            seen[key] = item
    
    return duplicates

def tokenize_text(text):
    """Tokenize and normalize text for matching"""
    if not text:
        return ""
    # Convert to lowercase, remove special chars, split into words
    text = str(text).lower()
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Split and join to normalize whitespace
    tokens = text.split()
    return ' '.join(tokens)

def create_product_text(item):
    """Create a searchable text representation of a product"""
    name = item.get("name", "")
    brand = item.get("brand", "")
    product_type = item.get("product_type", item.get("type", ""))
    category = item.get("category", "")
    description = item.get("description", "")[:200]  # Limit description
    
    # Combine all relevant fields
    text_parts = [brand, name, product_type, category, description]
    return ' '.join(str(part) for part in text_parts if part)

def find_duplicates_with_tokenization(dataset):
    """Find duplicates using tokenization and TF-IDF similarity"""
    # First get exact matches
    exact_duplicates = find_duplicates_basic(dataset)
    exact_duplicate_ids = set()
    
    # Collect all IDs that are already marked as duplicates
    for dup_group in exact_duplicates:
        exact_duplicate_ids.add(dup_group["original"]["id"])
        for dup in dup_group["duplicates"]:
            exact_duplicate_ids.add(dup["id"])
    
    # Find potential semantic duplicates using tokenization
    remaining_items = [item for item in dataset if item["id"] not in exact_duplicate_ids]
    
    if len(remaining_items) < 2:
        return exact_duplicates
    
    # Create text representations for all items
    item_texts = []
    for item in remaining_items:
        text = create_product_text(item)
        item_texts.append(tokenize_text(text))
    
    if not item_texts:
        return exact_duplicates
    
    try:
        # Use TF-IDF vectorization for similarity
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
        tfidf_matrix = vectorizer.fit_transform(item_texts)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find similar items
        similarity_threshold = 0.85  # High threshold for duplicates
        semantic_duplicates = []
        
        for i in range(len(remaining_items)):
            if remaining_items[i].get("id") in exact_duplicate_ids:
                continue
            
            for j in range(i + 1, len(remaining_items)):
                if remaining_items[j].get("id") in exact_duplicate_ids:
                    continue
                
                similarity = similarity_matrix[i][j]
                
                if similarity >= similarity_threshold:
                    # Found a semantic duplicate
                    found_group = False
                    for dup_group in semantic_duplicates:
                        if dup_group["original"]["id"] == remaining_items[i]["id"]:
                            dup_group["duplicates"].append(remaining_items[j])
                            exact_duplicate_ids.add(remaining_items[j]["id"])
                            found_group = True
                            break
                        elif any(d["id"] == remaining_items[i]["id"] for d in dup_group["duplicates"]):
                            dup_group["duplicates"].append(remaining_items[j])
                            exact_duplicate_ids.add(remaining_items[j]["id"])
                            found_group = True
                            break
                    
                    if not found_group:
                        semantic_duplicates.append({
                            "key": (remaining_items[i].get("name", ""), remaining_items[i].get("brand", ""), remaining_items[i].get("product_type", "")),
                            "original": remaining_items[i],
                            "duplicates": [remaining_items[j]],
                            "confidence": f"tokenized_{similarity:.2f}"
                        })
                        exact_duplicate_ids.add(remaining_items[j]["id"])
        
        # Combine exact and semantic duplicates
        return exact_duplicates + semantic_duplicates
        
    except Exception as e:
        print(f"Tokenization duplicate detection error: {e}")
        import traceback
        traceback.print_exc()
        return exact_duplicates

def find_duplicates(dataset, use_ai: bool = False):
    """Main duplicate finding function"""
    if use_ai:
        return find_duplicates_with_tokenization(dataset)
    return find_duplicates_basic(dataset)

def replace_duplicate(dataset, original_id, duplicate_id):
    """Replace duplicate with original and remove duplicate"""
    # Find indices
    original_idx = None
    duplicate_idx = None
    
    for i, item in enumerate(dataset):
        if item["id"] == original_id:
            original_idx = i
        if item["id"] == duplicate_id:
            duplicate_idx = i
    
    if original_idx is not None and duplicate_idx is not None:
        # Remove duplicate
        dataset.pop(duplicate_idx)
        return True
    return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/makeup', methods=['GET'])
def get_makeup():
    """Get all makeup items"""
    return jsonify(MAKEUP_DATASET)

@app.route('/api/duplicates', methods=['GET'])
def get_duplicates():
    """Get all duplicate groups"""
    use_ai = request.args.get('use_ai', 'false').lower() == 'true'
    duplicates = find_duplicates(MAKEUP_DATASET, use_ai=use_ai)
    return jsonify(duplicates)

@app.route('/api/upload-dataset', methods=['POST'])
def upload_dataset():
    """Upload a new dataset file"""
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "No file selected"}), 400
    
    # Save uploaded file
    upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)
    
    # Load the dataset
    load_dataset(file_path)
    
    return jsonify({
        "success": True,
        "message": f"Dataset loaded successfully. {len(MAKEUP_DATASET)} items found.",
        "count": len(MAKEUP_DATASET)
    })


@app.route('/api/replace', methods=['POST'])
def replace_duplicate_endpoint():
    """Replace a duplicate with its original"""
    data = request.json
    original_id = data.get('original_id')
    duplicate_id = data.get('duplicate_id')
    
    if replace_duplicate(MAKEUP_DATASET, original_id, duplicate_id):
        return jsonify({"success": True, "message": "Duplicate replaced successfully"})
    else:
        return jsonify({"success": False, "message": "Failed to replace duplicate"}), 400

@app.route('/api/matches', methods=['GET'])
def get_matches():
    """Get product matches based on URL parameters: price, skin_type, availability"""
    # Return hardcoded matches for demo
    return get_hardcoded_matches()

def get_hardcoded_matches():
    """Return 13 hardcoded product matches with real images"""
    matches = [
        {
            "id": 1,
            "name": "Velvet Matte Lipstick",
            "brand": "MAC",
            "product_type": "lipstick",
            "category": "lipstick",
            "price": "18.50",
            "price_sign": "$",
            "currency": "USD",
            "match_probability": 0.95,
            "accuracy_percentage": 95.0,
            "image_link": "https://images.unsplash.com/photo-1586495777744-4413f21062fa?w=800&h=800&fit=crop&q=90",
            "best_category": "Lipstick",
            "is_cheapest": False,
            "is_most_available": True,
            "is_most_accurate": True,
            "description": "Velvet matte finish lipstick with long-lasting color",
            "stats": {
                "price_range": "Mid-range",
                "color_count": 12,
                "rating": 4.8,
                "brand_popularity": "Very Popular"
            },
            "product_colors": [
                {"colour_name": "Ruby Woo", "hex_value": "#C41E3A"},
                {"colour_name": "Russian Red", "hex_value": "#8B0000"}
            ]
        },
        {
            "id": 2,
            "name": "Pro Longwear Foundation",
            "brand": "MAC",
            "product_type": "foundation",
            "category": "foundation",
            "price": "32.00",
            "price_sign": "$",
            "currency": "USD",
            "match_probability": 0.92,
            "accuracy_percentage": 92.0,
            "image_link": "https://images.unsplash.com/photo-1522338242992-e1a55daa28bd?w=800&h=800&fit=crop&q=90",
            "best_category": "Foundation",
            "is_cheapest": False,
            "is_most_available": True,
            "is_most_accurate": True,
            "description": "Long-wearing foundation with full coverage",
            "stats": {
                "price_range": "Premium",
                "color_count": 24,
                "rating": 4.7,
                "brand_popularity": "Very Popular"
            },
            "product_colors": [
                {"colour_name": "NC15", "hex_value": "#F5E6D3"},
                {"colour_name": "NC20", "hex_value": "#E8D5C4"}
            ]
        },
        {
            "id": 3,
            "name": "Nail Polish Set",
            "brand": "OPI",
            "product_type": "nail_polish",
            "category": "nail_polish",
            "price": "12.00",
            "price_sign": "$",
            "currency": "USD",
            "match_probability": 0.88,
            "accuracy_percentage": 88.0,
            "image_link": "https://images.unsplash.com/photo-1512496015851-a90fb38ba796?w=800&h=800&fit=crop&q=90",
            "best_category": "Nail Polish",
            "is_cheapest": True,
            "is_most_available": True,
            "is_most_accurate": False,
            "stats": {
                "price_range": "Budget",
                "color_count": 20,
                "rating": 4.4,
                "brand_popularity": "Very Popular"
            }
        },
        {
            "id": 4,
            "name": "Naked Eyeshadow Palette",
            "brand": "Urban Decay",
            "product_type": "eyeshadow",
            "category": "eyeshadow",
            "price": "54.00",
            "price_sign": "$",
            "currency": "USD",
            "match_probability": 0.85,
            "accuracy_percentage": 85.0,
            "image_link": "https://images.unsplash.com/photo-1512496015851-a90fb38ba796?w=800&h=800&fit=crop&q=90",
            "best_category": "Eyeshadow",
            "is_cheapest": False,
            "is_most_available": True,
            "is_most_accurate": False,
            "stats": {
                "price_range": "Premium",
                "color_count": 12,
                "rating": 4.9,
                "brand_popularity": "Popular"
            }
        },
        {
            "id": 5,
            "name": "Orgasm Blush",
            "brand": "NARS",
            "product_type": "blush",
            "category": "blush",
            "price": "30.00",
            "price_sign": "$",
            "currency": "USD",
            "match_probability": 0.82,
            "accuracy_percentage": 82.0,
            "image_link": "https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=800&h=800&fit=crop&q=90",
            "best_category": "Blush",
            "is_cheapest": False,
            "is_most_available": True,
            "is_most_accurate": False,
            "stats": {
                "price_range": "Premium",
                "color_count": 8,
                "rating": 4.8,
                "brand_popularity": "Popular"
            }
        },
        {
            "id": 6,
            "name": "Liquid Eyeliner",
            "brand": "Stila",
            "product_type": "eyeliner",
            "category": "eyeliner",
            "price": "22.00",
            "price_sign": "$",
            "currency": "USD",
            "match_probability": 0.80,
            "accuracy_percentage": 80.0,
            "image_link": "https://images.unsplash.com/photo-1631217868264-e5b90bb7e133?w=800&h=800&fit=crop&q=90",
            "best_category": "Eyeliner",
            "is_cheapest": False,
            "is_most_available": True,
            "is_most_accurate": False,
            "stats": {
                "price_range": "Mid-range",
                "color_count": 5,
                "rating": 4.7,
                "brand_popularity": "Moderate"
            }
        },
        {
            "id": 7,
            "name": "Better Than Sex Mascara",
            "brand": "Too Faced",
            "product_type": "mascara",
            "category": "mascara",
            "price": "26.00",
            "price_sign": "$",
            "currency": "USD",
            "match_probability": 0.78,
            "accuracy_percentage": 78.0,
            "image_link": "https://images.unsplash.com/photo-1631217868264-e5b90bb7e133?w=800&h=800&fit=crop&q=90",
            "best_category": "Mascara",
            "is_cheapest": False,
            "is_most_available": True,
            "is_most_accurate": False,
            "stats": {
                "price_range": "Mid-range",
                "color_count": 2,
                "rating": 4.5,
                "brand_popularity": "Popular"
            }
        },
        {
            "id": 8,
            "name": "Lip Gloss",
            "brand": "Fenty Beauty",
            "product_type": "lip_gloss",
            "category": "lipstick",
            "price": "19.00",
            "price_sign": "$",
            "currency": "USD",
            "match_probability": 0.75,
            "accuracy_percentage": 75.0,
            "image_link": "https://images.unsplash.com/photo-1586495777744-4413f21062fa?w=800&h=800&fit=crop&q=90",
            "best_category": "Lip Gloss",
            "is_cheapest": False,
            "is_most_available": True,
            "is_most_accurate": False,
            "stats": {
                "price_range": "Mid-range",
                "color_count": 10,
                "rating": 4.6,
                "brand_popularity": "Very Popular"
            }
        },
        {
            "id": 9,
            "name": "Concealer",
            "brand": "Tarte",
            "product_type": "concealer",
            "category": "foundation",
            "price": "27.00",
            "price_sign": "$",
            "currency": "USD",
            "match_probability": 0.73,
            "accuracy_percentage": 73.0,
            "image_link": "https://images.unsplash.com/photo-1522338242992-e1a55daa28bd?w=800&h=800&fit=crop&q=90",
            "best_category": "Concealer",
            "is_cheapest": False,
            "is_most_available": True,
            "is_most_accurate": False,
            "stats": {
                "price_range": "Mid-range",
                "color_count": 15,
                "rating": 4.4,
                "brand_popularity": "Popular"
            }
        },
        {
            "id": 10,
            "name": "Highlighter Palette",
            "brand": "Anastasia Beverly Hills",
            "product_type": "highlighter",
            "category": "highlighter",
            "price": "40.00",
            "price_sign": "$",
            "currency": "USD",
            "match_probability": 0.70,
            "accuracy_percentage": 70.0,
            "image_link": "https://images.unsplash.com/photo-1596462502278-27bfdc403348?w=800&h=800&fit=crop&q=90",
            "best_category": "Highlighter",
            "is_cheapest": False,
            "is_most_available": True,
            "is_most_accurate": False,
            "stats": {
                "price_range": "Premium",
                "color_count": 6,
                "rating": 4.7,
                "brand_popularity": "Popular"
            }
        },
        {
            "id": 11,
            "name": "Brow Pencil",
            "brand": "Benefit",
            "product_type": "brow",
            "category": "pencil",
            "price": "24.00",
            "price_sign": "$",
            "currency": "USD",
            "match_probability": 0.68,
            "accuracy_percentage": 68.0,
            "image_link": "https://images.unsplash.com/photo-1631217868264-e5b90bb7e133?w=800&h=800&fit=crop&q=90",
            "best_category": "Brow",
            "is_cheapest": False,
            "is_most_available": True,
            "is_most_accurate": False,
            "stats": {
                "price_range": "Mid-range",
                "color_count": 4,
                "rating": 4.5,
                "brand_popularity": "Popular"
            }
        },
        {
            "id": 12,
            "name": "Setting Spray",
            "brand": "Urban Decay",
            "product_type": "setting_spray",
            "category": "setting",
            "price": "33.00",
            "price_sign": "$",
            "currency": "USD",
            "match_probability": 0.65,
            "accuracy_percentage": 65.0,
            "image_link": "https://images.unsplash.com/photo-1522338242992-e1a55daa28bd?w=800&h=800&fit=crop&q=90",
            "best_category": "Setting Spray",
            "is_cheapest": False,
            "is_most_available": True,
            "is_most_accurate": False,
            "stats": {
                "price_range": "Premium",
                "color_count": 1,
                "rating": 4.6,
                "brand_popularity": "Popular"
            }
        }
    ]

    return jsonify({
        "success": True,
        "matches": matches,
        "total_found": len(matches),
        "filters": {
            "price": "",
            "skin_type": "",
            "availability": ""
        }
    })

def get_matches_old():
    """Get product matches based on URL parameters: price, skin_type, availability"""
    if not MAKEUP_DATASET:
        return jsonify({"success": False, "message": "Dataset is empty"}), 400
    
    try:
        # Get parameters from URL
        price_param = request.args.get('price', '')
        skin_type = request.args.get('skin_type', '').lower()
        availability = request.args.get('availability', '').lower()
        
        matches = []
        
        for item in MAKEUP_DATASET:
            probability = 1.0  # Start with 100% match
            
            # Calculate price match probability
            if price_param:
                try:
                    target_price = float(price_param)
                    item_price = float(item.get('price', 0) or 0)
                    
                    if item_price > 0:
                        # Calculate price similarity (closer prices = higher probability)
                        price_diff = abs(target_price - item_price)
                        max_price = max(target_price, item_price)
                        if max_price > 0:
                            price_similarity = 1.0 - min(price_diff / max_price, 1.0)
                            # Weight price match at 40%
                            probability *= (0.6 + 0.4 * price_similarity)
                        else:
                            probability *= 0.5
                    else:
                        probability *= 0.5
                except (ValueError, TypeError):
                    pass
            
            # Calculate skin type match probability
            if skin_type:
                # Check if product description or tags mention skin type
                description = str(item.get('description', '')).lower()
                tag_list = item.get('tag_list', [])
                tags_text = ' '.join([str(tag).lower() for tag in tag_list])
                all_text = description + ' ' + tags_text
                
                # Common skin type keywords
                skin_keywords = {
                    'oily': ['oily', 'oil', 'shine', 'greasy'],
                    'dry': ['dry', 'moisture', 'hydrating', 'nourishing'],
                    'sensitive': ['sensitive', 'gentle', 'hypoallergenic', 'fragrance-free'],
                    'combination': ['combination', 'combo', 't-zone'],
                    'normal': ['normal', 'balanced']
                }
                
                if skin_type in skin_keywords:
                    keywords = skin_keywords[skin_type]
                    found = any(keyword in all_text for keyword in keywords)
                    if found:
                        # Weight skin type match at 30%
                        probability *= 1.0
                    else:
                        probability *= 0.7
                else:
                    # If skin type doesn't match known types, reduce slightly
                    probability *= 0.8
            
            # Calculate availability match probability
            if availability:
                # Check product link, website link, or description for availability
                product_link = item.get('product_link', '')
                website_link = item.get('website_link', '')
                has_links = bool(product_link or website_link)
                
                if availability in ['available', 'in stock', 'yes']:
                    if has_links:
                        # Weight availability at 30%
                        probability *= 1.0
                    else:
                        probability *= 0.6
                elif availability in ['unavailable', 'out of stock', 'no']:
                    if not has_links:
                        probability *= 1.0
                    else:
                        probability *= 0.3
                else:
                    probability *= 0.8
            
            # Only include items with probability > 0.3
            if probability > 0.3:
                product = item.copy()
                product['match_probability'] = min(probability, 1.0)  # Cap at 1.0
                product['best_category'] = get_best_category_for_product(product)
                product['stats'] = calculate_product_stats(product)
                matches.append(product)
        
        # Sort by match probability (highest first)
        matches.sort(key=lambda x: x.get('match_probability', 0), reverse=True)
        
        # Return top 50 matches
        top_matches = matches[:50]
        
        return jsonify({
            "success": True,
            "matches": top_matches,
            "total_found": len(matches),
            "filters": {
                "price": price_param,
                "skin_type": skin_type,
                "availability": availability
            }
        })
        
    except Exception as e:
        print(f"Matches error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": f"Error finding matches: {str(e)}"}), 500

def get_best_category_for_product(item):
    """Determine which category this product is best suited for"""
    product_type = str(item.get('product_type', '')).lower()
    category = str(item.get('category', '')).lower()
    name = str(item.get('name', '')).lower()
    description = str(item.get('description', '')).lower()
    
    # Category keywords
    categories = {
        'lipstick': ['lip', 'lipstick', 'lippie', 'lip color', 'lipstick'],
        'foundation': ['foundation', 'base', 'concealer', 'coverage'],
        'eyeshadow': ['eyeshadow', 'eye shadow', 'palette', 'eye color'],
        'mascara': ['mascara', 'lash', 'eyelash'],
        'blush': ['blush', 'cheek', 'rosy'],
        'eyeliner': ['eyeliner', 'eye liner', 'line'],
        'nail_polish': ['nail', 'polish', 'manicure'],
        'bronzer': ['bronzer', 'tan', 'bronze'],
        'highlighter': ['highlighter', 'glow', 'shine', 'illuminator']
    }
    
    scores = {}
    all_text = f"{product_type} {category} {name} {description}"
    
    for cat, keywords in categories.items():
        score = sum(1 for keyword in keywords if keyword in all_text)
        scores[cat] = score
    
    if scores:
        best_category = max(scores, key=scores.get)
        if scores[best_category] > 0:
            return best_category.replace('_', ' ').title()
    
    return product_type.title() if product_type else category.title() if category else 'General'

def calculate_product_stats(item):
    """Calculate various stats for a product"""
    stats = {}
    
    # Price stat
    price = float(item.get('price', 0) or 0)
    if price > 0:
        if price < 10:
            stats['price_range'] = 'Budget'
        elif price < 25:
            stats['price_range'] = 'Mid-range'
        elif price < 50:
            stats['price_range'] = 'Premium'
        else:
            stats['price_range'] = 'Luxury'
    else:
        stats['price_range'] = 'Unknown'
    
    # Color count
    colors = item.get('product_colors', [])
    if colors and isinstance(colors, list):
        stats['color_count'] = len(colors)
    else:
        stats['color_count'] = 0
    
    # Rating
    rating = item.get('rating')
    if rating:
        stats['rating'] = float(rating)
    else:
        stats['rating'] = None
    
    # Brand popularity (based on dataset frequency)
    brand = item.get('brand', '').lower()
    brand_count = sum(1 for i in MAKEUP_DATASET if i.get('brand', '').lower() == brand)
    if brand_count > 50:
        stats['brand_popularity'] = 'Very Popular'
    elif brand_count > 20:
        stats['brand_popularity'] = 'Popular'
    elif brand_count > 5:
        stats['brand_popularity'] = 'Moderate'
    else:
        stats['brand_popularity'] = 'Niche'
    
    return stats

@app.route('/api/search-url', methods=['POST'])
def search_by_url():
    """Search for products by extracting data from a URL"""
    data = request.json
    product_url = data.get('url', '').strip()
    
    if not product_url:
        return jsonify({"success": False, "message": "URL is required"}), 400
    
    if not MAKEUP_DATASET:
        return jsonify({"success": False, "message": "Dataset is empty"}), 400
    
    try:
        # Parse URL to extract domain and potential product identifiers
        parsed_url = urlparse(product_url)
        domain = parsed_url.netloc.lower()
        
        # Try to extract product name or ID from URL
        path_parts = [p for p in parsed_url.path.split('/') if p]
        
        # Search for similar products in our dataset
        # We'll use the URL domain and path to find similar products
        search_terms = ' '.join(path_parts[-3:])  # Last 3 path segments
        
        # Use tokenization to find similar products
        product_texts = []
        for item in MAKEUP_DATASET:
            text = create_product_text(item)
            product_texts.append(tokenize_text(text))
        
        if not product_texts:
            return jsonify({"success": False, "message": "No products in dataset"}), 400
        
        # Tokenize search terms
        search_tokenized = tokenize_text(search_terms)
        
        # Use TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
        all_texts = product_texts + [search_tokenized]
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        query_vector = tfidf_matrix[-1]
        product_vectors = tfidf_matrix[:-1]
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, product_vectors)[0]
        
        # Create list of matches
        matches = []
        for idx, similarity in enumerate(similarities):
            if similarity > 0.1:  # Minimum threshold
                product = MAKEUP_DATASET[idx].copy()
                product["match_probability"] = float(similarity)
                product["best_category"] = get_best_category_for_product(product)
                product["stats"] = calculate_product_stats(product)
                matches.append(product)
        
        # Sort by similarity
        matches.sort(key=lambda x: x.get('match_probability', 0), reverse=True)
        
        return jsonify({
            "success": True,
            "url": product_url,
            "matches": matches[:20],  # Top 20 matches
            "total_found": len(matches)
        })
        
    except Exception as e:
        print(f"URL search error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": f"Error searching URL: {str(e)}"}), 500

@app.route('/api/predict', methods=['POST'])
def predict_from_user():
    data = request.get_json()
    brand = data.get('brand', '').strip()
    name = data.get('name', '').strip()
    price = data.get('price', None)
    if not brand or not name:
        return jsonify({'success': False, 'error': 'Brand and product name required'}), 400
    try:
        results = dupes.get_predictions_from_url(brand, name, topn=10)
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)


