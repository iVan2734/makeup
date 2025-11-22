# Makeup Duplicate Finder

A web application to find and replace duplicate makeup items in a dataset using tokenization and TF-IDF similarity matching.

## Features

- 📁 **Upload Your Dataset**: Support for JSON and CSV files
- 🔍 **Tokenization-Based Search**: Type a product name and find matching products using TF-IDF tokenization
- 🎯 **Smart Duplicate Detection**: Find duplicates based on name, brand, color, and type
- 🤖 **Tokenization-Powered Matching**: Advanced tokenization for semantic duplicate detection
- 🎨 **Beautiful UI**: Modern, responsive interface with tabbed navigation
- ✅ **Easy Replacement**: One-click duplicate replacement

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

### Uploading Your Dataset

1. Click "Choose File" in the upload section
2. Select a JSON or CSV file with your makeup data
3. The dataset will be automatically loaded

### Dataset Format

**JSON Format:**
```json
[
    {"id": 1, "name": "Red Lipstick", "brand": "MAC", "color": "Red", "type": "Lipstick", "price": 18.50},
    {"id": 2, "name": "Nude Foundation", "brand": "Maybelline", "color": "Beige", "type": "Foundation", "price": 12.99}
]
```

**CSV Format:**
```csv
id,name,brand,color,type,price
1,Red Lipstick,MAC,Red,Lipstick,18.50
2,Nude Foundation,Maybelline,Beige,Foundation,12.99
```

The app will automatically detect field names (case-insensitive) like:
- `name` or `Name`
- `brand` or `Brand`
- `color` or `Color`
- `type`, `Type`, `category`, or `Category`
- `price`, `Price`, `cost`, or `Cost`
- `id`, `ID`, or `Id`

### Using Tokenization Features

1. No API keys needed! The app uses local tokenization and TF-IDF for all matching
2. Check "Use tokenization for smarter duplicate detection" to enable advanced duplicate matching
3. All features work offline using the data from your JSON file

### Searching for Products

1. Go to the **🔍 Search Products** tab
2. Type a product name (e.g., "Red Lipstick", "Foundation", "Mascara")
3. Click "Search" or press Enter
4. View results sorted by similarity score (tokenization-based matching)
5. Results show match percentage and all product details

### Finding and Replacing Duplicates

1. Go to the **Duplicates** tab
2. Review duplicate groups (original items are highlighted in green, duplicates in red)
3. Click **Replace** on any duplicate to remove it and keep the original
4. The dataset will update automatically

## API Endpoints

- `GET /api/makeup` - Get all makeup items
- `POST /api/search` - Search for products using tokenization (no API key needed)
- `GET /api/duplicates?use_ai=true` - Get duplicate groups (optional tokenization matching)
- `POST /api/replace` - Replace a duplicate with original
- `POST /api/upload-dataset` - Upload a new dataset file

## Notes

- Duplicates are identified by matching: name, brand, color, and type
- The app handles various field name formats automatically
- Uses TF-IDF tokenization for semantic matching - no external APIs required
- All processing is done locally using your dataset


