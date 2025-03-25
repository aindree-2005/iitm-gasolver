import os
import zipfile
import pandas as pd
import httpx
import json
import shutil
import tempfile
from typing import Dict, Any, List, Optional
import re
import tempfile
import shutil
import subprocess
import httpx
import json
import csv
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

async def execute_command(command: str) -> str:
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error executing command: {str(e)}"

async def count_tokens(text: str) -> str:
    """Counts tokens in a message sent to OpenAI API."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer dummy_api_key",
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 1,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            prompt_tokens = response.json().get("usage", {}).get("prompt_tokens", 0)

            return f"Token Count: {prompt_tokens} tokens"
    except Exception as e:
        return f"Error: {e}"

async def extract_zip_and_read_csv(
    file_path: str, column_name: Optional[str] = None
) -> str:
    """
    Extract a zip file and read a value from a CSV file inside it
    """
    temp_dir = tempfile.mkdtemp()

    try:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        csv_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv")]

        if not csv_files:
            return "No CSV files found in the zip file."
        csv_path = os.path.join(temp_dir, csv_files[0])
        df = pd.read_csv(csv_path)
        if column_name and column_name in df.columns:
            return str(df[column_name].iloc[0])
        elif "answer" in df.columns:
            return str(df["answer"].iloc[0])
        else:
            return f"CSV contains columns: {', '.join(df.columns)}"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


async def extract_zip_and_process_files(file_path: str, operation: str) -> str:
    """
    Extract a zip file and process multiple files
    """
    temp_dir = tempfile.mkdtemp()

    try:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        if operation == "find_different_lines":
            file_a = os.path.join(temp_dir, "a.txt")
            file_b = os.path.join(temp_dir, "b.txt")

            if not os.path.exists(file_a) or not os.path.exists(file_b):
                return "Files a.txt and b.txt not found."

            with open(file_a, "r") as a, open(file_b, "r") as b:
                a_lines = a.readlines()
                b_lines = b.readlines()

                diff_count = sum(
                    1
                    for i in range(min(len(a_lines), len(b_lines)))
                    if a_lines[i] != b_lines[i]
                )
                return str(diff_count)

        elif operation == "count_large_files":
            large_file_count = 0
            threshold = 1024 * 1024  # 1MB in bytes

            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    if file_size > threshold:
                        large_file_count += 1

            return str(large_file_count)

        elif operation == "count_files_by_extension":
            extension_counts = {}

            for root, _, files in os.walk(temp_dir):
                for file in files:
                    _, ext = os.path.splitext(file)
                    ext = ext.lower()
                    extension_counts[ext] = extension_counts.get(ext, 0) + 1

            return json.dumps(extension_counts)

        elif operation == "list":
            file_list = []
            for root, dirs, files in os.walk(temp_dir):
                rel_path = os.path.relpath(root, temp_dir)
                if rel_path == ".":
                    rel_path = ""
                for dir_name in dirs:
                    dir_path = (
                        os.path.join(rel_path, dir_name) if rel_path else dir_name
                    )
                    file_list.append(f"📁 {dir_path}/")

                # Add files with sizes
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    file_size = os.path.getsize(file_path)

                    # Format size
                    if file_size < 1024:
                        size_str = f"{file_size} B"
                    elif file_size < 1024 * 1024:
                        size_str = f"{file_size/1024:.1f} KB"
                    else:
                        size_str = f"{file_size/(1024*1024):.1f} MB"

                    file_rel_path = (
                        os.path.join(rel_path, file_name) if rel_path else file_name
                    )
                    file_list.append(f"📄 {file_rel_path} ({size_str})")
            if not file_list:
                return "The zip file is empty."

            return "Contents of the zip file:\n\n" + "\n".join(file_list)

        else:
            return f"Unsupported operation: {operation}"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
async def merge_csv_files(zip_path: str, merge_column: str) -> str:
    temp_dir = tempfile.mkdtemp()
    
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        csv_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".csv")]
        if not csv_files:
            return "No CSV files found in the ZIP archive."

        dfs = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                if merge_column in df.columns:
                    dfs.append(df)
                else:
                    return f"Column '{merge_column}' not found in {os.path.basename(file)}"
            except Exception as e:
                return f"Error reading {os.path.basename(file)}: {e}"

        if not dfs:
            return "No valid CSV files with the specified column."

        merged_df = pd.concat(dfs, ignore_index=True)
        result_path = os.path.join(temp_dir, "merged_result.csv")
        merged_df.to_csv(result_path, index=False)

        return f"Merged {len(dfs)} CSV files. Result has {len(merged_df)} rows and {len(merged_df.columns)} columns."
    
    except Exception as e:
        return f"Error merging CSV files: {e}"

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
def count_days_of_week(start_date: str, end_date: str, day_of_week: str) -> str:
    """
    Count occurrences of a specific day of the week between two dates

    Args:
        start_date: Start date in ISO format (YYYY-MM-DD)
        end_date: End date in ISO format (YYYY-MM-DD)
        day_of_week: Day of the week to count

    Returns:
        Count of the specified day of the week
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        day_map = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6,
        }
        weekday = day_map.get(day_of_week)
        if weekday is None:
            return f"Invalid day of week: {day_of_week}"
        count = 0
        current = start
        while current <= end:
            if current.weekday() == weekday:
                count += 1
            current += timedelta(days=1)

        return str(count)
    except Exception as e:
        return f"Error counting days of week: {str(e)}"
import pandas as pd
import re
import shutil
import tempfile
import zipfile
from datetime import datetime
import dateutil.parser

async def clean_sales_data_and_calculate_margin(file_path: str, cutoff_date_str: str, product_filter: str, country_filter: str) -> str:
    try:
        # Parse cutoff date
        try:
            date_match = re.search(r"([A-Za-z]+ [A-Za-z]+ \d+ \d+ \d+:\d+:\d+)", cutoff_date_str)
            cutoff_date = datetime.strptime(date_match.group(1), "%a %b %d %Y %H:%M:%S") if date_match else dateutil.parser.parse(cutoff_date_str)
        except Exception as e:
            return f"Error parsing date: {e}"

        # Read Excel file
        df = pd.read_excel(file_path)
        col_map = {
            "customer": ["customer", "client", "buyer"],
            "country": ["country", "nation", "region"],
            "date": ["date", "transaction date", "sale date"],
            "product": ["product", "item", "goods"],
            "sales": ["sales", "revenue", "amount"],
            "cost": ["cost", "expense", "purchase price"],
        }
        
        # Map columns
        mapped_cols = {key: next((col for col in df.columns if col.lower() in [x.lower() for x in names]), None) for key, names in col_map.items()}
        if any(v is None for k, v in mapped_cols.items() if k in ["date", "product", "country", "sales", "cost"]):
            return "Error: Missing required columns."

        df = df.rename(columns=mapped_cols)

        # Standardize country names
        country_map = {"usa": "US", "united states": "US", "uk": "UK", "united kingdom": "UK", "france": "FR", "brazil": "BR", "india": "IN"}
        df["country"] = df["country"].str.strip().str.lower().map(lambda x: country_map.get(x, x.upper()))

        # Parse dates
        df["date"] = df["date"].apply(lambda x: dateutil.parser.parse(str(x)) if pd.notna(x) else None)

        # Clean numeric columns
        df["sales"] = pd.to_numeric(df["sales"].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")
        df["cost"] = pd.to_numeric(df["cost"].astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")
        df["cost"].fillna(df["sales"] * 0.5, inplace=True)

        # Filter data
        df = df[(df["date"] <= cutoff_date) & (df["product"].str.lower() == product_filter.lower()) & (df["country"].str.lower() == country_filter.lower())]

        if df.empty:
            return "0.0000"

        margin = (df["sales"].sum() - df["cost"].sum()) / df["sales"].sum() if df["sales"].sum() else 0
        return f"{margin:.4f}"
    
    except Exception as e:
        return f"Error processing sales data: {e}"
async def analyze_sentiment(text: str, api_key: str = "dummy_api_key") -> str:
    """
    Analyze sentiment of text using OpenAI API
    """
    import httpx
    import json

    url = "https://api.openai.com/v1/chat/completions"

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "Analyze the sentiment of the following text and classify it as GOOD, BAD, or NEUTRAL.",
            },
            {"role": "user", "content": text},
        ],
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()

            # Extract the sentiment analysis result
            sentiment = result["choices"][0]["message"]["content"]

            return f"""
# Sentiment Analysis Result

## Input Text

## Analysis
{sentiment}

## API Request Details
- Model: gpt-4o-mini
- API Endpoint: {url}
- Request Type: POST
"""
    except Exception as e:
        return f"Error analyzing sentiment: {str(e)}"
async def calculate_statistics(file_path: str, operation: str, column_name: str) -> str:
    """
    Calculate statistics from a CSV file.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Verify that the column exists
        if column_name not in df.columns:
            return f"Column '{column_name}' not found in the CSV file."

        # Perform the requested operation
        if operation == "sum":
            result = df[column_name].sum()
        elif operation == "average":
            result = df[column_name].mean()
        elif operation == "median":
            result = df[column_name].median()
        elif operation == "max":
            result = df[column_name].max()
        elif operation == "min":
            result = df[column_name].min()
        else:
            return f"Unsupported operation: {operation}"

        return str(result)

    except Exception as e:
        return f"Error calculating statistics: {str(e)}"


def sort_json_array(json_array: str, sort_keys: list) -> str:
    """
    Sort a JSON array based on specified criteria

    Args:
        json_array: JSON array as a string
        sort_keys: List of keys to sort by

    Returns:
        Sorted JSON array as a string
    """
    try:
        # Parse the JSON array
        data = json.loads(json_array)

        # Sort the data based on the specified keys
        for key in reversed(sort_keys):
            data = sorted(data, key=lambda x: x.get(key, ""))

        # Return the sorted JSON as a string without whitespace
        return json.dumps(data, separators=(",", ":"))

    except Exception as e:
        return f"Error sorting JSON array: {str(e)}"

async def find_most_similar_phrases(embeddings_dict: Dict[str, List[float]]) -> str:
    """
    Find the most similar pair of phrases based on cosine similarity of their embeddings

    Args:
        embeddings_dict: Dictionary mapping phrases to their embeddings

    Returns:
        The most similar pair of phrases
    """
    try:
        import numpy as np
        from itertools import combinations

        # Function to calculate cosine similarity
        def cosine_similarity(vec1, vec2):
            dot_product = np.dot(vec1, vec2)
            norm_vec1 = np.linalg.norm(vec1)
            norm_vec2 = np.linalg.norm(vec2)
            return dot_product / (norm_vec1 * norm_vec2)

        # Convert dictionary to lists for easier processing
        phrases = list(embeddings_dict.keys())
        embeddings = list(embeddings_dict.values())

        # Calculate similarity for each pair
        max_similarity = -1
        most_similar_pair = None

        for i, j in combinations(range(len(phrases)), 2):
            similarity = cosine_similarity(embeddings[i], embeddings[j])
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_pair = (phrases[i], phrases[j])

        # Generate Python code for the solution
        solution_code = """
def most_similar(embeddings):
    \"\"\"
    Find the most similar pair of phrases based on cosine similarity of their embeddings.
    
    Args:
        embeddings: Dictionary mapping phrases to their embeddings
        
    Returns:
        Tuple of the two most similar phrases
    \"\"\"
    import numpy as np
    from itertools import combinations

    # Function to calculate cosine similarity
    def cosine_similarity(vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

    # Convert dictionary to lists for easier processing
    phrases = list(embeddings.keys())
    embeddings_list = list(embeddings.values())

    # Calculate similarity for each pair
    max_similarity = -1
    most_similar_pair = None

    for i, j in combinations(range(len(phrases)), 2):
        similarity = cosine_similarity(embeddings_list[i], embeddings_list[j])
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_pair = (phrases[i], phrases[j])

    return most_similar_pair
"""

        return f"""
# Most Similar Phrases Analysis

## Result
The most similar pair of phrases is: {most_similar_pair[0]} and {most_similar_pair[1]}
Similarity score: {max_similarity:.4f}

## Python Solution
```python
{solution_code}
```

## Explanation
This function:

1. Calculates the cosine similarity between each pair of embeddings
2. Identifies the pair with the highest similarity score
3. Returns the two phrases as a tuple
"""
    except Exception as e:
        return f"Error finding most similar phrases: {str(e)}"
async def count_json_key_occurrences(file_path: str, target_key: str) -> str:
    """
    Count occurrences of a specific key in a nested JSON structure

    Args:
        file_path: Path to the JSON file
        target_key: The key to search for in the JSON structure

    Returns:
        Count of occurrences of the target key
    """
    try:
        import json

        # Load the JSON file
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Initialize counter
        count = 0

        # Define a recursive function to traverse the JSON structure
        def traverse_json(obj):
            nonlocal count

            if isinstance(obj, dict):
                # Check keys at this level
                for key in obj:
                    if key == target_key:
                        count += 1
                    # Recursively check values that are objects or arrays
                    traverse_json(obj[key])
            elif isinstance(obj, list):
                # Recursively check each item in the array
                for item in obj:
                    traverse_json(item)

        # Start traversal
        traverse_json(data)

        # Return just the count as a string
        return str(count)

    except Exception as e:
        import traceback

        return (
            f"Error counting JSON key occurrences: {str(e)}\n{traceback.format_exc()}"
        )


async def reconstruct_scrambled_image(
    image_path: str, mapping_data: str, output_path: str = None
) -> str:
    """
    Reconstruct an image from scrambled pieces using a mapping

    Args:
        image_path: Path to the scrambled image
        mapping_data: String containing the mapping data (tab or space separated)
        output_path: Path to save the reconstructed image (optional)

    Returns:
        Path to the reconstructed image or error message
    """
    try:
        import os
        import tempfile
        from PIL import Image
        import numpy as np
        import re

        # Load the scrambled image
        scrambled_image = Image.open(image_path)
        width, height = scrambled_image.size

        # Determine grid size (assuming square grid and pieces)
        # Parse the mapping data to get the grid dimensions
        mapping_lines = mapping_data.strip().split("\n")
        grid_size = 0

        # Find the maximum row and column values to determine grid size
        for line in mapping_lines:
            # Skip header line if present
            if re.match(r"^\D", line):  # Line starts with non-digit
                continue

            # Extract numbers from the line
            numbers = re.findall(r"\d+", line)
            if len(numbers) >= 4:  # Ensure we have enough values
                for num in numbers:
                    grid_size = max(
                        grid_size, int(num) + 1
                    )  # +1 because indices start at 0

        # Calculate piece dimensions
        piece_width = width // grid_size
        piece_height = height // grid_size

        # Create a mapping dictionary from the mapping data
        mapping = {}

        for line in mapping_lines:
            # Skip header line if present
            if re.match(r"^\D", line):
                continue

            # Extract numbers from the line
            numbers = re.findall(r"\d+", line)
            if len(numbers) >= 4:
                orig_row, orig_col, scram_row, scram_col = map(int, numbers[:4])
                mapping[(scram_row, scram_col)] = (orig_row, orig_col)

        # Create a new image for the reconstructed result
        reconstructed_image = Image.new("RGB", (width, height))

        # Place each piece in its original position
        for scram_pos, orig_pos in mapping.items():
            scram_row, scram_col = scram_pos
            orig_row, orig_col = orig_pos

            # Calculate pixel coordinates
            scram_x = scram_col * piece_width
            scram_y = scram_row * piece_height
            orig_x = orig_col * piece_width
            orig_y = orig_row * piece_height

            # Extract the piece from the scrambled image
            piece = scrambled_image.crop(
                (scram_x, scram_y, scram_x + piece_width, scram_y + piece_height)
            )

            # Place the piece in the reconstructed image
            reconstructed_image.paste(piece, (orig_x, orig_y))

        # Save the reconstructed image
        if output_path is None:
            # Create a temporary file if no output path is provided
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            output_path = temp_file.name
            temp_file.close()

        reconstructed_image.save(output_path, format="PNG")

        return output_path

    except Exception as e:
        import traceback

        return f"Error reconstructing image: {str(e)}\n{traceback.format_exc()}"
