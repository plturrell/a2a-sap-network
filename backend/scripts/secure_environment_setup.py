#!/usr/bin/env python3
"""
Secure Environment Setup Script for A2A Agents
Automatically generates secure configuration from template
"""

import os
import sys
import secrets
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SecureEnvironmentSetup:
    """Handles secure environment configuration setup"""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.project_root = Path(__file__).parent.parent
        self.template_path = self.project_root / ".env.template"
        self.env_path = self.project_root / ".env"
        self.secrets_dir = self.project_root / ".secrets"
        
        # Security configuration based on environment
        self.security_config = {
            "development": {
                "jwt_secret_length": 32,
                "password_length": 24,
                "api_key_length": 32,
                "require_ssl": False,
                "enable_debug": True
            },
            "staging": {
                "jwt_secret_length": 64,
                "password_length": 32,
                "api_key_length": 48,
                "require_ssl": True,
                "enable_debug": False
            },
            "production": {
                "jwt_secret_length": 64,
                "password_length": 40,
                "api_key_length": 64,
                "require_ssl": True,
                "enable_debug": False
            }
        }
    
    def generate_secure_value(self, value_type: str, length: int = 32) -> str:
        """Generate cryptographically secure values"""
        
        if value_type == "password":
            # Generate strong password with mixed characters
            alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*+-="
            return ''.join(secrets.choice(alphabet) for _ in range(length))
        
        elif value_type == "jwt_secret":
            # Generate URL-safe token for JWT
            return secrets.token_urlsafe(length)
        
        elif value_type == "api_key":
            # Generate hex token for API keys
            return secrets.token_hex(length)
        
        elif value_type == "private_key":
            # Generate 32-byte private key as hex (64 characters)
            return secrets.token_hex(32)
        
        elif value_type == "salt":
            # Generate salt for key derivation
            return secrets.token_hex(16)
        
        elif value_type == "agent_id":
            # Generate unique agent ID
            timestamp = str(int(datetime.utcnow().timestamp()))
            random_suffix = secrets.token_hex(8)
            return f"a2a-agent-{timestamp}-{random_suffix}"
        
        elif value_type == "webhook_secret":
            # Generate webhook secret
            return secrets.token_urlsafe(32)
        
        else:
            # Default to URL-safe token
            return secrets.token_urlsafe(length)
    
    def create_secrets_directory(self):
        """Create secure secrets directory"""
        try:
            self.secrets_dir.mkdir(mode=0o700, exist_ok=True)
            logger.info(f"Created secrets directory: {self.secrets_dir}")
            
            # Create .gitignore to ensure secrets aren't committed
            gitignore_path = self.secrets_dir / ".gitignore"
            with open(gitignore_path, 'w') as f:
                f.write("# Exclude all secrets from version control\n*\n!.gitignore\n")
            
        except Exception as e:
            logger.error(f"Failed to create secrets directory: {e}")
            raise
    
    def generate_master_key(self) -> str:
        """Generate master encryption key for secrets manager"""
        try:
            from cryptography.fernet import Fernet
            
            master_key = Fernet.generate_key()
            master_key_path = self.secrets_dir / "master.key"
            
            # Write with restrictive permissions
            with open(master_key_path, 'wb') as f:
                f.write(master_key)
            
            # Set file permissions to owner read/write only
            os.chmod(master_key_path, 0o600)
            
            logger.info(f"Generated master encryption key: {master_key_path}")
            return str(master_key_path)
            
        except ImportError:
            logger.warning("cryptography package not available, using fallback key generation")
            return str(self.secrets_dir / "master.key")
        except Exception as e:
            logger.error(f"Failed to generate master key: {e}")
            raise
    
    def generate_ssl_certificates(self) -> Dict[str, str]:
        """Generate self-signed SSL certificates for development"""
        if self.environment == "production":
            logger.warning("Self-signed certificates should not be used in production")
            return {}
        
        try:
            ssl_dir = self.secrets_dir / "ssl"
            ssl_dir.mkdir(exist_ok=True)
            
            cert_path = ssl_dir / "cert.pem"
            key_path = ssl_dir / "key.pem"
            
            # Generate self-signed certificate using OpenSSL
            cmd = [
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", str(key_path),
                "-out", str(cert_path),
                "-days", "365", "-nodes",
                "-subj", "/C=US/ST=State/L=City/O=A2A/CN=localhost"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Set restrictive permissions
                os.chmod(cert_path, 0o644)
                os.chmod(key_path, 0o600)
                
                logger.info("Generated SSL certificates for development")
                return {
                    "SSL_CERT_PATH": str(cert_path),
                    "SSL_KEY_PATH": str(key_path)
                }
            else:
                logger.warning(f"Failed to generate SSL certificates: {result.stderr}")
                return {}
                
        except FileNotFoundError:
            logger.warning("OpenSSL not found, skipping SSL certificate generation")
            return {}
        except Exception as e:
            logger.error(f"Error generating SSL certificates: {e}")
            return {}
    
    def get_database_credentials(self) -> Dict[str, str]:
        """Generate or prompt for database credentials"""
        
        if self.environment == "development":
            # Use default development credentials
            return {
                "DATABASE_URL": "postgresql://a2a_dev:a2a_dev_password@localhost:5432/a2a_dev",
                "PRIMARY_DATABASE_URL": "postgresql://a2a_dev:a2a_dev_password@localhost:5432/a2a_dev"
            }
        else:
            # For staging/production, generate secure credentials
            db_password = self.generate_secure_value("password", 24)
            db_user = f"a2a_{self.environment}"
            db_name = f"a2a_{self.environment}"
            
            logger.info(f"Generated database credentials for {self.environment}")
            logger.info(f"Database user: {db_user}")
            logger.info(f"Database password: {db_password}")
            logger.warning("Please create database and user manually with these credentials")
            
            return {
                "DATABASE_URL": f"postgresql://{db_user}:{db_password}@localhost:5432/{db_name}",
                "PRIMARY_DATABASE_URL": f"postgresql://{db_user}:{db_password}@localhost:5432/{db_name}"
            }
    
    def generate_environment_config(self) -> Dict[str, str]:
        """Generate complete environment configuration"""
        
        config = self.security_config.get(self.environment, self.security_config["development"])
        
        # Create secrets directory
        self.create_secrets_directory()
        
        # Generate master key
        master_key_path = self.generate_master_key()
        
        # Generate SSL certificates if needed
        ssl_config = self.generate_ssl_certificates()
        
        # Get database credentials
        db_config = self.get_database_credentials()
        
        # Generate all secure values
        secure_values = {
            # Basic security
            "A2A_MASTER_PASSWORD": self.generate_secure_value("password", config["password_length"]),
            "A2A_SALT": self.generate_secure_value("salt"),
            "JWT_SECRET": self.generate_secure_value("jwt_secret", config["jwt_secret_length"]),
            
            # Redis password
            "REDIS_PASSWORD": self.generate_secure_value("password", 20),
            
            # Blockchain
            "BLOCKCHAIN_PRIVATE_KEY": self.generate_secure_value("private_key"),
            
            # Agent identity
            "A2A_AGENT_ID": self.generate_secure_value("agent_id"),
            
            # Webhook security
            "WEBHOOK_SECRET": self.generate_secure_value("webhook_secret"),
            
            # Secrets manager
            "SECRETS_MASTER_KEY_PATH": master_key_path,
            
            # Environment-specific settings
            "NODE_ENV": self.environment,
            "DEBUG_ENABLED": str(config["enable_debug"]).lower(),
            "SSL_ENABLED": str(config["require_ssl"]).lower(),
            "DEPLOYMENT_ENVIRONMENT": self.environment,
            
            # Security settings
            "SECURITY_ENABLED": "true",
            "RATE_LIMIT_ENABLED": "true",
            "AUDIT_ENABLED": "true",
            "MONITORING_ENABLED": "true",
        }
        
        # Add database config
        secure_values.update(db_config)
        
        # Add SSL config if available
        secure_values.update(ssl_config)
        
        # Add Redis URL with authentication
        if config["require_ssl"]:
            secure_values["REDIS_URL"] = f"rediss://:{secure_values['REDIS_PASSWORD']}@localhost:6379/0"
        else:
            secure_values["REDIS_URL"] = f"redis://:{secure_values['REDIS_PASSWORD']}@localhost:6379/0"
        
        return secure_values
    
    def create_environment_file(self, force: bool = False) -> bool:
        """Create .env file from template with secure values"""
        
        if self.env_path.exists() and not force:
            logger.error(f"Environment file already exists: {self.env_path}")
            logger.info("Use --force to overwrite existing file")
            return False
        
        if not self.template_path.exists():
            logger.error(f"Template file not found: {self.template_path}")
            return False
        
        try:
            # Read template
            with open(self.template_path, 'r') as f:
                template_content = f.read()
            
            # Generate secure configuration
            secure_config = self.generate_environment_config()
            
            # Replace placeholders in template
            env_content = template_content
            
            # Replace all REPLACE_WITH_* placeholders
            replacements = {
                "REPLACE_WITH_STRONG_RANDOM_PASSWORD_64_CHARS_MIN": secure_config["A2A_MASTER_PASSWORD"],
                "REPLACE_WITH_STRONG_RANDOM_SALT_32_CHARS_MIN": secure_config["A2A_SALT"],
                "REPLACE_WITH_CRYPTOGRAPHICALLY_SECURE_SECRET_64_CHARS_MIN": secure_config["JWT_SECRET"],
                "REPLACE_WITH_STRONG_REDIS_PASSWORD": secure_config["REDIS_PASSWORD"],
                "REPLACE_WITH_SECURE_GENERATED_PRIVATE_KEY_64_HEX_CHARS": secure_config["BLOCKCHAIN_PRIVATE_KEY"],
                "REPLACE_WITH_UNIQUE_AGENT_ID": secure_config["A2A_AGENT_ID"],
                "REPLACE_WITH_STRONG_WEBHOOK_SECRET": secure_config["WEBHOOK_SECRET"],
                
                # Database placeholders
                "USERNAME:PASSWORD@localhost:5432/a2a_dev": secure_config["DATABASE_URL"].split("postgresql://")[1],
                
                # Placeholder API keys (users must replace these manually)
                "REPLACE_WITH_ACTUAL_OPENAI_API_KEY": "sk-YOUR_OPENAI_API_KEY_HERE",
                "REPLACE_WITH_ACTUAL_ANTHROPIC_API_KEY": "sk-ant-YOUR_ANTHROPIC_API_KEY_HERE",
                "REPLACE_WITH_ACTUAL_XAI_API_KEY": "your-xai-api-key-here",
                "REPLACE_WITH_ACTUAL_GROK_API_KEY": "your-xai-api-key-here",
                
                # Contract addresses (deployment-specific)
                "0xREPLACE_WITH_ACTUAL_CONTRACT_ADDRESS": "0x0000000000000000000000000000000000000000",
            }
            
            for placeholder, value in replacements.items():
                env_content = env_content.replace(placeholder, value)
            
            # Apply environment-specific overrides
            for key, value in secure_config.items():
                # Replace the line that starts with this key
                import re


# A2A Protocol Compliance: All imports must be available
# No fallback implementations allowed - the agent must have all required dependencies
                pattern = f"^{key}=.*$"
                replacement = f"{key}={value}"
                env_content = re.sub(pattern, replacement, env_content, flags=re.MULTILINE)
            
            # Write environment file with restrictive permissions
            with open(self.env_path, 'w') as f:
                f.write(env_content)
            
            # Set restrictive permissions
            os.chmod(self.env_path, 0o600)
            
            logger.info(f"Created secure environment file: {self.env_path}")
            
            # Create backup for reference
            backup_path = self.env_path.with_suffix(f'.{self.environment}.backup')
            with open(backup_path, 'w') as f:
                f.write(env_content)
            os.chmod(backup_path, 0o600)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create environment file: {e}")
            return False
    
    def validate_environment(self) -> bool:
        """Validate the created environment configuration"""
        
        if not self.env_path.exists():
            logger.error("Environment file does not exist")
            return False
        
        try:
            with open(self.env_path, 'r') as f:
                content = f.read()
            
            # Check for remaining placeholders
            dangerous_patterns = [
                "REPLACE_WITH_",
                "YOUR_API_KEY_HERE",
                "test_pass",
                "default",
                "placeholder"
            ]
            
            issues = []
            for pattern in dangerous_patterns:
                if pattern in content:
                    issues.append(f"Found placeholder/insecure value: {pattern}")
            
            # Check for proper permissions
            file_stat = os.stat(self.env_path)
            file_mode = oct(file_stat.st_mode)[-3:]
            if file_mode != "600":
                issues.append(f"Environment file has insecure permissions: {file_mode}")
            
            if issues:
                logger.error("Environment validation failed:")
                for issue in issues:
                    logger.error(f"  - {issue}")
                return False
            else:
                logger.info("Environment validation passed")
                return True
                
        except Exception as e:
            logger.error(f"Environment validation error: {e}")
            return False
    
    def generate_security_checklist(self) -> str:
        """Generate post-setup security checklist"""
        
        checklist = f"""
# A2A Agents Security Checklist - {self.environment.upper()} Environment

## Generated Files
- ✓ Environment file: {self.env_path}
- ✓ Secrets directory: {self.secrets_dir}
- ✓ Master encryption key: {self.secrets_dir}/master.key

## Manual Tasks Required

### 1. API Keys Configuration
Replace placeholder API keys with actual credentials:
- [ ] OpenAI API key (OPENAI_API_KEY)
- [ ] Anthropic API key (ANTHROPIC_API_KEY)  
- [ ] XAI/Grok API key (XAI_API_KEY, GROK_API_KEY)

### 2. Database Setup
- [ ] Create database user and database
- [ ] Test database connection
- [ ] Configure SSL/TLS for database connections (production)

### 3. Redis Configuration
- [ ] Configure Redis authentication
- [ ] Enable Redis SSL/TLS (production)
- [ ] Test Redis connection

### 4. Smart Contract Deployment
- [ ] Deploy A2A smart contracts to blockchain
- [ ] Update contract addresses in environment file
- [ ] Verify contract functionality

### 5. SSL/TLS Certificates (Production)
- [ ] Obtain valid SSL certificates
- [ ] Update SSL_CERT_PATH and SSL_KEY_PATH
- [ ] Configure reverse proxy (if applicable)

### 6. Monitoring Setup
- [ ] Configure log aggregation
- [ ] Set up metrics collection
- [ ] Configure alerting rules
- [ ] Test health check endpoints

### 7. Security Hardening
- [ ] Run security scan: npm run security:scan
- [ ] Test authentication flow
- [ ] Verify rate limiting
- [ ] Test backup and recovery procedures

### 8. Final Validation
- [ ] Run: python scripts/secure_environment_setup.py --validate
- [ ] Load test the application
- [ ] Verify all integrations work
- [ ] Document any environment-specific configurations

## Security Notes
- Environment file permissions: 600 (owner read/write only)
- Secrets directory permissions: 700 (owner access only)
- Never commit .env file to version control
- Rotate secrets regularly in production
- Monitor for security vulnerabilities
"""
        return checklist.strip()


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="Secure Environment Setup for A2A Agents")
    parser.add_argument(
        "--environment", "-e",
        choices=["development", "staging", "production"],
        default="development",
        help="Target environment"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing environment file"
    )
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Validate existing environment configuration"
    )
    parser.add_argument(
        "--checklist", "-c",
        action="store_true",
        help="Generate security checklist"
    )
    
    args = parser.parse_args()
    
    setup = SecureEnvironmentSetup(args.environment)
    
    if args.validate:
        # Validate existing environment
        if setup.validate_environment():
            logger.info("Environment validation successful")
            sys.exit(0)
        else:
            logger.error("Environment validation failed")
            sys.exit(1)
    
    elif args.checklist:
        # Generate security checklist
        checklist = setup.generate_security_checklist()
        print(checklist)
        
        # Save checklist to file
        checklist_path = setup.project_root / f"SECURITY_CHECKLIST_{args.environment.upper()}.md"
        with open(checklist_path, 'w') as f:
            f.write(checklist)
        
        logger.info(f"Security checklist saved to: {checklist_path}")
        sys.exit(0)
    
    else:
        # Create environment configuration
        logger.info(f"Setting up secure environment for: {args.environment}")
        
        if setup.create_environment_file(args.force):
            logger.info("Environment setup completed successfully")
            
            # Validate the created environment
            if setup.validate_environment():
                logger.info("Environment validation passed")
                
                # Generate and display checklist
                checklist = setup.generate_security_checklist()
                print("\n" + "="*80)
                print(checklist)
                print("="*80)
                
                # Save checklist
                checklist_path = setup.project_root / f"SECURITY_CHECKLIST_{args.environment.upper()}.md"
                with open(checklist_path, 'w') as f:
                    f.write(checklist)
                
                logger.info(f"Security checklist saved to: {checklist_path}")
                logger.info("Complete the manual tasks in the checklist before deployment")
                
                sys.exit(0)
            else:
                logger.error("Environment validation failed")
                sys.exit(1)
        else:
            logger.error("Environment setup failed")
            sys.exit(1)


if __name__ == "__main__":
    main()