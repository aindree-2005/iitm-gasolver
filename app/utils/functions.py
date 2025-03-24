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

