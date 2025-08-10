#!/usr/bin/env python3
"""
Duplicate Component Cleanup Script
Safely removes duplicated components from a2aAgents after migration to a2aNetwork
"""

import os
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base paths
A2A_AGENTS_PATH = Path("/Users/apple/projects/a2a/a2aAgents/backend/app/a2a")
A2A_NETWORK_PATH = Path("/Users/apple/projects/a2a/a2aNetwork")

# Components to remove (duplicated in a2aNetwork)
COMPONENTS_TO_REMOVE = [
    # SDK components (duplicated in a2aNetwork/sdk/)
    "sdk/agentBase.py",
    "sdk/client.py", 
    "sdk/decorators.py",
    "sdk/types.py",
    "sdk/utils.py",
    
    # Security components (duplicated in a2aNetwork/trustSystem/)
    "security/delegationContracts.py",
    "security/sharedTrust.py", 
    "security/smartContractTrust.py"
]

# Keep these files (integration wrappers, not duplicates)
KEEP_FILES = [
    "sdk/__init__.py",  # Import management
    "security/__init__.py",  # Import management
    "network/",  # Integration layer
    "version/",  # Versioning system
]


def verify_a2a_network_exists():
    """Verify a2aNetwork components exist before cleanup"""
    required_paths = [
        A2A_NETWORK_PATH / "sdk" / "agentBase.py",
        A2A_NETWORK_PATH / "sdk" / "types.py",
        A2A_NETWORK_PATH / "trustSystem" / "smartContractTrust.py",
        A2A_NETWORK_PATH / "api" / "networkClient.py"
    ]
    
    for path in required_paths:
        if not path.exists():
            logger.error(f"Required a2aNetwork component missing: {path}")
            return False
            
    logger.info("✅ All required a2aNetwork components exist")
    return True


def backup_components():
    """Create backup of components before removal"""
    backup_dir = A2A_AGENTS_PATH / "backup_before_cleanup"
    backup_dir.mkdir(exist_ok=True)
    
    for component in COMPONENTS_TO_REMOVE:
        source = A2A_AGENTS_PATH / component
        if source.exists():
            # Create backup directory structure
            backup_target = backup_dir / component
            backup_target.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file to backup
            shutil.copy2(source, backup_target)
            logger.info(f"Backed up: {component}")
    
    logger.info(f"✅ Backup created at: {backup_dir}")
    return backup_dir


def remove_duplicate_components():
    """Remove duplicate components from a2aAgents"""
    removed_count = 0
    
    for component in COMPONENTS_TO_REMOVE:
        source = A2A_AGENTS_PATH / component
        if source.exists():
            source.unlink()
            logger.info(f"Removed duplicate: {component}")
            removed_count += 1
        else:
            logger.warning(f"Component not found: {component}")
    
    logger.info(f"✅ Removed {removed_count} duplicate components")
    return removed_count


def update_init_files():
    """Update __init__.py files to use conditional imports"""
    
    # Update SDK __init__.py
    sdk_init = A2A_AGENTS_PATH / "sdk" / "__init__.py"
    if sdk_init.exists():
        with open(sdk_init, 'w') as f:
            f.write('"""\nA2A Agent SDK\nSimplifies development of new agents in the A2A network\n"""\n\n# Import from a2aNetwork (preferred) or use local fallback\ntry:\n    import sys\n    sys.path.insert(0, "/Users/apple/projects/a2a/a2aNetwork")\n    from sdk.agentBase import A2AAgentBase\n    from sdk.client import A2AClient\n    from sdk.decorators import a2a_handler, a2a_task, a2a_skill\n    from sdk.types import A2AMessage, MessagePart, MessageRole\n    from sdk.types import TaskStatus, AgentCard, AgentCapability, SkillDefinition\n    from sdk.utils import create_agent_id, validate_message, sign_message\n    print("✅ Using a2aNetwork SDK components")\nexcept ImportError as e:\n    print(f"⚠️  a2aNetwork SDK not available: {e}")\n    raise ImportError("SDK components not available from a2aNetwork")\n\n__version__ = "1.0.0"\n__all__ = [\n    "A2AAgentBase",\n    "A2AClient", \n    "a2a_handler",\n    "a2a_task",\n    "a2a_skill",\n    "A2AMessage",\n    "MessagePart", \n    "MessageRole",\n    "TaskStatus",\n    "AgentCard",\n    "AgentCapability",\n    "SkillDefinition",\n    "create_agent_id",\n    "validate_message",\n    "sign_message"\n]\n')
        logger.info("Updated sdk/__init__.py for network imports")
    
    # Update Security __init__.py
    security_init = A2A_AGENTS_PATH / "security" / "__init__.py"
    if security_init.exists():
        with open(security_init, 'w') as f:
            f.write('"""\nA2A Security Components\nDelegated to a2aNetwork trust system\n"""\n\n# Import from a2aNetwork trust system\ntry:\n    import sys\n    sys.path.insert(0, "/Users/apple/projects/a2a/a2aNetwork")\n    from trustSystem.smartContractTrust import *\n    from trustSystem.delegationContracts import *\n    from trustSystem.sharedTrust import *\n    print("✅ Using a2aNetwork trust system")\nexcept ImportError as e:\n    print(f"⚠️  a2aNetwork trust system not available: {e}")\n    raise ImportError("Trust system not available from a2aNetwork")\n')
        logger.info("Updated security/__init__.py for network imports")


def verify_agent_imports():
    """Verify agent import statements are compatible"""
    agents_dir = A2A_AGENTS_PATH / "agents"
    agent_dirs = [d for d in agents_dir.iterdir() if d.is_dir() and not d.name.startswith('_')]
    
    compatible_count = 0
    for agent_dir in agent_dirs:
        active_dir = agent_dir / "active"
        if active_dir.exists():
            sdk_files = list(active_dir.glob("*Sdk.py"))
            for sdk_file in sdk_files:
                with open(sdk_file, 'r') as f:
                    content = f.read()
                    if "from app.a2a.sdk import" in content or "sys.path.insert(0, '/Users/apple/projects/a2a/a2aNetwork')" in content:
                        compatible_count += 1
                        logger.info(f"✅ {sdk_file.name} has compatible imports")
                    else:
                        logger.warning(f"⚠️  {sdk_file.name} may need import updates")
    
    logger.info(f"✅ Verified {compatible_count} agent files have compatible imports")
    return compatible_count


def main():
    """Main cleanup process"""
    logger.info("🧹 Starting duplicate component cleanup...")
    
    # Step 1: Verify a2aNetwork exists
    if not verify_a2a_network_exists():
        logger.error("❌ Cannot proceed - a2aNetwork components missing")
        return False
    
    # Step 2: Create backup
    backup_dir = backup_components()
    
    # Step 3: Remove duplicate components
    removed_count = remove_duplicate_components()
    
    # Step 4: Update __init__.py files
    update_init_files()
    
    # Step 5: Verify agent compatibility
    compatible_agents = verify_agent_imports()
    
    logger.info(f"""\n🎉 Cleanup completed successfully!\n\nSummary:\n- Removed {removed_count} duplicate components\n- Updated 2 __init__.py files\n- Verified {compatible_agents} agent import compatibility\n- Backup created at: {backup_dir}\n\nNext steps:\n1. Test all agents with: pytest tests/\n2. Run integration tests: aiq run --config_file workflow.yaml\n3. Verify network connectivity works\n""")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
