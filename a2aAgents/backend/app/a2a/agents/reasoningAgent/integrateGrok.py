"""
Integrate Grok with Reasoning Agent
Simple script to update reasoning agent to use GrokReasoning
"""

import os
import re
from pathlib import Path
from datetime import datetime

def integrate_grok():
    """Update reasoning agent to use GrokReasoning"""

    file_path = Path(__file__).parent / "reasoningAgent.py"

    if not file_path.exists():
        print(f"❌ {file_path} not found")
        return False

    # Backup
    backup = file_path.with_suffix(f'.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    file_path.rename(backup)
    print(f"📁 Backup created: {backup}")

    try:
        with open(backup, 'r') as f:
            content = f.read()

        # Replace groq import with GrokReasoning
        content = re.sub(
            r"from groq import Groq",
            "from .grokReasoning import GrokReasoning",
            content
        )

        # Update GROK_AVAILABLE check
        content = re.sub(
            r"try:\s*\n\s*from groq import Groq\s*\n\s*GROK_AVAILABLE = True\s*\nexcept ImportError:\s*\n\s*GROK_AVAILABLE = False",
            """try:
    from .grokReasoning import GrokReasoning
    GROK_AVAILABLE = True
except ImportError:
    GROK_AVAILABLE = False""",
            content
        )

        # Replace Groq() with GrokReasoning()
        content = re.sub(
            r"self\.grok_client = Groq\([^)]*\)",
            "self.grok_client = GrokReasoning()",
            content
        )

        # Update API calls to use new methods
        replacements = [
            # Old Groq API call -> New GrokReasoning method
            (r"self\.grok_client\.chat\.completions\.create\([^)]+\)",
             "await self.grok_client.decompose_question(prompt)"),

            # Response handling
            (r"response\.choices\[0\]\.message\.content",
             "response.get('decomposition', {})"),

            # Model reference
            (r'"llama3-groq-70b-8192-tool-use-preview"',
             '"grok-4-latest"')
        ]

        for old, new in replacements:
            content = re.sub(old, new, content)

        # Write updated file
        with open(file_path, 'w') as f:
            f.write(content)

        print("✅ Integration complete")
        print("\nChanges made:")
        print("- Replaced groq import with GrokReasoning")
        print("- Updated client initialization")
        print("- Updated API calls to use new methods")
        print(f"\nOriginal file backed up to: {backup}")

        return True

    except Exception as e:
        print(f"❌ Integration failed: {e}")
        backup.rename(file_path)  # Restore
        print("✅ Restored from backup")
        return False

if __name__ == "__main__":
    print("Integrating GrokReasoning with reasoning agent...\n")

    # Check for API key
    api_key = os.getenv('XAI_API_KEY') or os.getenv('GROK_API_KEY')
    if api_key:
        print(f"✅ API key found: {api_key[:20]}...")
    else:
        print("⚠️  No API key found. Add XAI_API_KEY to .env")

    print()
    integrate_grok()
