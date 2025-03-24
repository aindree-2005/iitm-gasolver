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
