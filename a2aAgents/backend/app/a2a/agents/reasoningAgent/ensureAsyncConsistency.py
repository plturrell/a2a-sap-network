"""
Ensure Async Consistency
Updates all components to use async versions consistently
"""

import os
import re
from pathlib import Path


def update_imports_to_async():
    """Update all imports to use async versions"""
    replacements = [
        # Memory system
        (r'from \.reasoningMemorySystem import', 'from .asyncReasoningMemorySystem import'),
        (r'ReasoningMemorySystem\(', 'AsyncReasoningMemorySystem('),

        # SQLite
        (r'import sqlite3', 'import aiosqlite'),
        (r'sqlite3\.connect', 'aiosqlite.connect'),

        # File operations
        (r'with open\(', 'async with aiofiles.open('),
        (r'\.read\(\)', '.read()'),
        (r'\.write\(', '.write('),
    ]

    # Files to update (excluding tests for now)
    files_to_update = [
        'reasoningAgent.py',
        'enhancedReasoningAgent.py',
        'mcpReasoningAgent.py',
        'blackboardArchitecture.py',
    ]

    for filename in files_to_update:
        filepath = Path(filename)
        if not filepath.exists():
            continue

        print(f"Updating {filename} for async consistency...")

        with open(filepath, 'r') as f:
            content = f.read()

        original = content
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)

        if content != original:
            # Backup original
            backup_path = filepath.with_suffix('.backup.async')
            with open(backup_path, 'w') as f:
                f.write(original)

            # Write updated
            with open(filepath, 'w') as f:
                f.write(content)

            print(f"  ✅ Updated {filename}")
        else:
            print(f"  ℹ️ No changes needed in {filename}")


def add_async_memory_to_agent():
    """Add async memory system to reasoning agent if not present"""
    agent_file = Path('reasoningAgent.py')

    if not agent_file.exists():
        return

    with open(agent_file, 'r') as f:
        content = f.read()

    # Check if async memory is already imported
    if 'asyncReasoningMemorySystem' in content:
        print("Async memory already imported")
        return

    # Add import after other imports
    import_line = "from .asyncReasoningMemorySystem import AsyncReasoningMemorySystem\n"

    # Find where to insert (after other local imports)
    lines = content.split('\n')
    insert_index = 0

    for i, line in enumerate(lines):
        if line.startswith('from .') and 'import' in line:
            insert_index = i + 1

    lines.insert(insert_index, import_line)

    # Add initialization in __init__
    init_code = """
        # Initialize async memory system
        self.memory_system = None  # Will be initialized in initialize()
"""

    # Find __init__ method
    for i, line in enumerate(lines):
        if 'def __init__' in line and 'self' in line:
            # Find end of __init__
            j = i + 1
            while j < len(lines) and (lines[j].startswith('        ') or not lines[j].strip()):
                j += 1
            lines.insert(j - 1, init_code)
            break

    # Add async initialization
    init_method = """
    async def _initialize_memory_system(self):
        \"\"\"Initialize async memory system\"\"\"
        try:
            self.memory_system = AsyncReasoningMemorySystem()
            await self.memory_system.initialize()
            logger.info("Async memory system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
"""

    # Add before other initialization methods
    for i, line in enumerate(lines):
        if 'async def initialize' in line:
            lines.insert(i - 1, init_method)
            break

    # Update content
    content = '\n'.join(lines)

    # Write back
    with open(agent_file, 'w') as f:
        f.write(content)

    print("✅ Added async memory system to reasoning agent")


if __name__ == "__main__":
    print("Ensuring Async Consistency")
    print("=" * 50)

    # Update imports
    update_imports_to_async()

    # Add async memory
    add_async_memory_to_agent()

    print("\n✅ Async consistency updates complete!")
