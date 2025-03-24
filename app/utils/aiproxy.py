import os
import httpx
import json
import re
import zipfile
import pandas as pd
import tempfile
import shutil
import subprocess
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from app.utils.functions import *

load_dotenv()

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
AIPROXY_BASE_URL = "https://aiproxy.sanand.workers.dev/openai/v1"

async def get_openai_response(question: str, file_path: Optional[str] = None) -> str:
    """
    Get response from OpenAI via AI Proxy
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
    }
    functions = [
        {
            "type": "function",
            "function": {
                "name": "execute_command",
                "description": "Execute a shell command and return its output",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute",
                        }
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "extract_zip_and_read_csv",
                "description": "Extract a zip file and read a value from a CSV file inside it",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the zip file",
                        },
                        "column_name": {
                            "type": "string",
                            "description": "Column name to extract value from",
                        },
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "extract_zip_and_process_files",
                "description": "Extract a zip file and process multiple files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the zip file",
                        },
                        "operation": {
                            "type": "string",
                            "description": "Operation to perform on files",
                        },
                    },
                    "required": ["file_path", "operation"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "count_days_of_week",
                "description": "Count occurrences of a specific day of the week between two dates",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_date": {
                            "type": "string",
                            "description": "Start date in ISO format (YYYY-MM-DD)",
                        },
                        "end_date": {
                            "type": "string",
                            "description": "End date in ISO format (YYYY-MM-DD)",
                        },
                        "day_of_week": {
                            "type": "string",
                            "enum": [
                                "Monday",
                                "Tuesday",
                                "Wednesday",
                                "Thursday",
                                "Friday",
                                "Saturday",
                                "Sunday",
                            ],
                            "description": "Day of the week to count",
                        },
                    },
                    "required": ["start_date", "end_date", "day_of_week"],
                },
            },
        },
    ]

    messages = [
        {
            "role": "system",
            "content": "You are an assistant designed to solve data science assignment problems. You should use the provided functions when appropriate to solve the problem.",
        },
        {"role": "user", "content": question},
    ]

    # Add information about the file if provided
    if file_path:
        messages.append(
            {
                "role": "user",
                "content": f"I've uploaded a file that you can process. The file is stored at: {file_path}",
            }
        )

    # Prepare the request payload
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "tools": functions,
        "tool_choice": "auto",
    }

    # Make the request to the AI Proxy
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{AIPROXY_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60.0,
        )

        if response.status_code != 200:
            raise Exception(f"Error from OpenAI API: {response.text}")

        result = response.json()
        answer = None

        # Process the response
        message = result["choices"][0]["message"]

        # Check if there's a function call
        if "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])

                # Execute the appropriate function
                if function_name == "execute_command":
                    answer = await execute_command(function_args.get("command"))
        
        if answer is None:
            answer = message.get("content", "No response received.")
        
        return answer
