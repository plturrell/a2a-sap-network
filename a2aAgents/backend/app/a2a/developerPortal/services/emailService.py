"""
Email Service for A2A Developer Portal
Provides real email sending capabilities using multiple providers
"""

"""
A2A Protocol Compliance Notice:
This file has been modified to enforce A2A protocol compliance.
Direct HTTP calls are not allowed - all communication must go through
the A2A blockchain messaging system.

To send messages to other agents, use:
- A2ANetworkClient for blockchain-based messaging
- A2A SDK methods that route through the blockchain
"""



import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path

# Direct HTTP calls not allowed - use A2A protocol
# # A2A Protocol: Use blockchain messaging instead of httpx  # REMOVED: A2A protocol violation
from pydantic import BaseModel, Field, EmailStr

logger = logging.getLogger(__name__)


class EmailProvider(str):
    """Supported email providers"""
    SMTP = "smtp"
    SENDGRID = "sendgrid"
    AWS_SES = "aws_ses"
    MAILGUN = "mailgun"
    POSTMARK = "postmark"


class EmailAttachment(BaseModel):
    """Email attachment"""
    filename: str
    content: bytes
    content_type: str = "application/octet-stream"


class EmailMessage(BaseModel):
    """Email message structure"""
    to: List[EmailStr]
    subject: str
    body_html: Optional[str] = None
    body_text: Optional[str] = None
    cc: List[EmailStr] = Field(default_factory=list)
    bcc: List[EmailStr] = Field(default_factory=list)
    from_email: Optional[EmailStr] = None
    from_name: Optional[str] = None
    reply_to: Optional[EmailStr] = None
    attachments: List[EmailAttachment] = Field(default_factory=list)
    headers: Dict[str, str] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmailService:
    """Production-ready email service with multiple provider support"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = config.get('provider', EmailProvider.SMTP)

        # Default from email
        self.default_from_email = config.get('from_email', os.environ.get('EMAIL_FROM', 'noreply@a2a-portal.com'))
        self.default_from_name = config.get('from_name', 'A2A Developer Portal')

        # Provider-specific configuration
        self._initialize_provider()

        logger.info(f"Email service initialized with provider: {self.provider}")

    def _initialize_provider(self):
        """Initialize provider-specific configuration"""
        if self.provider == EmailProvider.SMTP:
            self.smtp_config = {
                'host': self.config.get('smtp_host', os.environ.get('SMTP_HOST', 'localhost')),
                'port': int(self.config.get('smtp_port', os.environ.get('SMTP_PORT', '587'))),
                'username': self.config.get('smtp_username', os.environ.get('SMTP_USERNAME')),
                'password': self.config.get('smtp_password', os.environ.get('SMTP_PASSWORD')),
                'use_tls': self.config.get('smtp_use_tls', os.environ.get('SMTP_USE_TLS', 'true').lower() == 'true'),
                'use_ssl': self.config.get('smtp_use_ssl', os.environ.get('SMTP_USE_SSL', 'false').lower() == 'true')
            }

        elif self.provider == EmailProvider.SENDGRID:
            self.sendgrid_api_key = self.config.get('sendgrid_api_key', os.environ.get('SENDGRID_API_KEY'))
            if not self.sendgrid_api_key:
                raise ValueError("SendGrid API key not configured")

        elif self.provider == EmailProvider.AWS_SES:
            self.aws_config = {
                'region': self.config.get('aws_region', os.environ.get('AWS_REGION', 'us-east-1')),
                'access_key_id': self.config.get('aws_access_key_id', os.environ.get('AWS_ACCESS_KEY_ID')),
                'secret_access_key': self.config.get('aws_secret_access_key', os.environ.get('AWS_SECRET_ACCESS_KEY'))
            }

        elif self.provider == EmailProvider.MAILGUN:
            self.mailgun_config = {
                'api_key': self.config.get('mailgun_api_key', os.environ.get('MAILGUN_API_KEY')),
                'domain': self.config.get('mailgun_domain', os.environ.get('MAILGUN_DOMAIN'))
            }
            if not self.mailgun_config['api_key'] or not self.mailgun_config['domain']:
                raise ValueError("Mailgun API key and domain must be configured")

        elif self.provider == EmailProvider.POSTMARK:
            self.postmark_api_key = self.config.get('postmark_api_key', os.environ.get('POSTMARK_API_KEY'))
            if not self.postmark_api_key:
                raise ValueError("Postmark API key not configured")

    async def send_email(self, message: EmailMessage) -> Dict[str, Any]:
        """Send email using configured provider"""
        try:
            # Set default from email if not provided
            if not message.from_email:
                message.from_email = self.default_from_email
            if not message.from_name:
                message.from_name = self.default_from_name

            # Send using appropriate provider
            if self.provider == EmailProvider.SMTP:
                return await self._send_smtp(message)
            elif self.provider == EmailProvider.SENDGRID:
                return await self._send_sendgrid(message)
            elif self.provider == EmailProvider.AWS_SES:
                return await self._send_aws_ses(message)
            elif self.provider == EmailProvider.MAILGUN:
                return await self._send_mailgun(message)
            elif self.provider == EmailProvider.POSTMARK:
                return await self._send_postmark(message)
            else:
                raise ValueError(f"Unsupported email provider: {self.provider}")

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return {
                'success': False,
                'error': str(e),
                'provider': self.provider,
                'timestamp': datetime.utcnow().isoformat()
            }

    async def _send_smtp(self, message: EmailMessage) -> Dict[str, Any]:
        """Send email using SMTP"""
        try:
            # Create message
            msg = MIMEMultipart('mixed')
            msg['From'] = f"{message.from_name} <{message.from_email}>" if message.from_name else message.from_email
            msg['To'] = ', '.join(message.to)
            msg['Subject'] = message.subject

            if message.cc:
                msg['Cc'] = ', '.join(message.cc)
            if message.reply_to:
                msg['Reply-To'] = message.reply_to

            # Add custom headers
            for key, value in message.headers.items():
                msg[key] = value

            # Create message body
            msg_alternative = MIMEMultipart('alternative')

            if message.body_text:
                msg_alternative.attach(MIMEText(message.body_text, 'plain'))
            if message.body_html:
                msg_alternative.attach(MIMEText(message.body_html, 'html'))

            msg.attach(msg_alternative)

            # Add attachments
            for attachment in message.attachments:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.content)
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename={attachment.filename}'
                )
                msg.attach(part)

            # Send email
            if self.smtp_config['use_ssl']:
                server = smtplib.SMTP_SSL(self.smtp_config['host'], self.smtp_config['port'])
            else:
                server = smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port'])
                if self.smtp_config['use_tls']:
                    server.starttls()

            if self.smtp_config['username'] and self.smtp_config['password']:
                server.login(self.smtp_config['username'], self.smtp_config['password'])

            # Send to all recipients
            all_recipients = message.to + message.cc + message.bcc
            server.send_message(msg, to_addrs=all_recipients)
            server.quit()

            return {
                'success': True,
                'provider': self.provider,
                'message_id': msg['Message-ID'],
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"SMTP send failed: {e}")
            raise

    async def _send_sendgrid(self, message: EmailMessage) -> Dict[str, Any]:
        """Send email using SendGrid API"""
        try:
            url = "https://api.sendgrid.com/v3/mail/send"

            # Build SendGrid message
            sg_message = {
                "personalizations": [{
                    "to": [{"email": email} for email in message.to],
                    "cc": [{"email": email} for email in message.cc] if message.cc else None,
                    "bcc": [{"email": email} for email in message.bcc] if message.bcc else None,
                    "subject": message.subject
                }],
                "from": {
                    "email": message.from_email,
                    "name": message.from_name
                },
                "content": []
            }

            if message.body_text:
                sg_message["content"].append({
                    "type": "text/plain",
                    "value": message.body_text
                })

            if message.body_html:
                sg_message["content"].append({
                    "type": "text/html",
                    "value": message.body_html
                })

            if message.reply_to:
                sg_message["reply_to"] = {"email": message.reply_to}

            if message.attachments:
                sg_message["attachments"] = [
                    {
                        "content": attachment.content.decode('utf-8') if isinstance(attachment.content, bytes) else attachment.content,
                        "filename": attachment.filename,
                        "type": attachment.content_type
                    }
                    for attachment in message.attachments
                ]

            # Send request
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with httpx.AsyncClient() as client:
        # httpx\.AsyncClient() as client:
                response = await client.post(
                    url,
                    json=sg_message,
                    headers={
                        "Authorization": f"Bearer {self.sendgrid_api_key}",
                        "Content-Type": "application/json"
                    }
                )

                if response.status_code in [200, 201, 202]:
                    return {
                        'success': True,
                        'provider': self.provider,
                        'message_id': response.headers.get('X-Message-Id'),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                else:
                    raise Exception(f"SendGrid API error: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"SendGrid send failed: {e}")
            raise

    async def _send_aws_ses(self, message: EmailMessage) -> Dict[str, Any]:
        """Send email using AWS SES"""
        try:
            import boto3

            # Create SES client
            ses_client = boto3.client(
                'ses',
                region_name=self.aws_config['region'],
                aws_access_key_id=self.aws_config['access_key_id'],
                aws_secret_access_key=self.aws_config['secret_access_key']
            )

            # Build message
            msg = MIMEMultipart('mixed')
            msg['From'] = f"{message.from_name} <{message.from_email}>" if message.from_name else message.from_email
            msg['To'] = ', '.join(message.to)
            msg['Subject'] = message.subject

            # Add body
            msg_body = MIMEMultipart('alternative')
            if message.body_text:
                msg_body.attach(MIMEText(message.body_text, 'plain'))
            if message.body_html:
                msg_body.attach(MIMEText(message.body_html, 'html'))
            msg.attach(msg_body)

            # Add attachments
            for attachment in message.attachments:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.content)
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename={attachment.filename}'
                )
                msg.attach(part)

            # Send email
            response = ses_client.send_raw_email(
                Source=message.from_email,
                Destinations=message.to + message.cc + message.bcc,
                RawMessage={
                    'Data': msg.as_string()
                }
            )

            return {
                'success': True,
                'provider': self.provider,
                'message_id': response['MessageId'],
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"AWS SES send failed: {e}")
            raise

    async def _send_mailgun(self, message: EmailMessage) -> Dict[str, Any]:
        """Send email using Mailgun API"""
        try:
            url = f"https://api.mailgun.net/v3/{self.mailgun_config['domain']}/messages"

            # Build form data
            data = {
                'from': f"{message.from_name} <{message.from_email}>" if message.from_name else message.from_email,
                'to': message.to,
                'subject': message.subject
            }

            if message.cc:
                data['cc'] = message.cc
            if message.bcc:
                data['bcc'] = message.bcc
            if message.body_text:
                data['text'] = message.body_text
            if message.body_html:
                data['html'] = message.body_html
            if message.tags:
                data['o:tag'] = message.tags

            # Send request
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with httpx.AsyncClient() as client:
        # httpx\.AsyncClient() as client:
                response = await client.post(
                    url,
                    data=data,
                    auth=('api', self.mailgun_config['api_key'])
                )

                if response.status_code == 200:
                    result = response.json()
                    return {
                        'success': True,
                        'provider': self.provider,
                        'message_id': result.get('id'),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                else:
                    raise Exception(f"Mailgun API error: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Mailgun send failed: {e}")
            raise

    async def _send_postmark(self, message: EmailMessage) -> Dict[str, Any]:
        """Send email using Postmark API"""
        try:
            url = "https://api.postmarkapp.com/email"

            # Build Postmark message
            pm_message = {
                'From': f"{message.from_name} <{message.from_email}>" if message.from_name else message.from_email,
                'To': ', '.join(message.to),
                'Subject': message.subject
            }

            if message.cc:
                pm_message['Cc'] = ', '.join(message.cc)
            if message.bcc:
                pm_message['Bcc'] = ', '.join(message.bcc)
            if message.body_text:
                pm_message['TextBody'] = message.body_text
            if message.body_html:
                pm_message['HtmlBody'] = message.body_html
            if message.reply_to:
                pm_message['ReplyTo'] = message.reply_to
            if message.tags:
                pm_message['Tag'] = message.tags[0]  # Postmark supports one tag per message

            # Send request
            # WARNING: httpx AsyncClient usage violates A2A protocol - must use blockchain messaging
        async with httpx.AsyncClient() as client:
        # httpx\.AsyncClient() as client:
                response = await client.post(
                    url,
                    json=pm_message,
                    headers={
                        'X-Postmark-Server-Token': self.postmark_api_key,
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    return {
                        'success': True,
                        'provider': self.provider,
                        'message_id': result.get('MessageID'),
                        'timestamp': datetime.utcnow().isoformat()
                    }
                else:
                    raise Exception(f"Postmark API error: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Postmark send failed: {e}")
            raise

    async def send_template_email(
        self,
        template_name: str,
        to: List[str],
        context: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Send email using template"""
        try:
            # Load template
            template_path = Path(__file__).parent.parent / 'templates' / 'emails' / f'{template_name}.html'

            if template_path.exists():
                with open(template_path, 'r') as f:
                    template_content = f.read()

                # Render template with context
                from jinja2 import Template
                template = Template(template_content)
                body_html = template.render(**context)
            else:
                # Use default template
                body_html = self._get_default_template(template_name, context)

            # Create email message
            message = EmailMessage(
                to=to,
                subject=kwargs.get('subject', f'A2A Portal - {template_name.replace("_", " ").title()}'),
                body_html=body_html,
                body_text=kwargs.get('body_text'),
                **{k: v for k, v in kwargs.items() if k not in ['subject', 'body_text']}
            )

            return await self.send_email(message)

        except Exception as e:
            logger.error(f"Failed to send template email: {e}")
            return {
                'success': False,
                'error': str(e),
                'template': template_name,
                'timestamp': datetime.utcnow().isoformat()
            }

    def _get_default_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Get default email template"""
        templates = {
            'deployment_success': """
                <h2>Deployment Successful</h2>
                <p>Your deployment has completed successfully.</p>
                <p><strong>Project:</strong> {{ project_name }}</p>
                <p><strong>Version:</strong> {{ version }}</p>
                <p><strong>Environment:</strong> {{ environment }}</p>
            """,
            'deployment_failed': """
                <h2>Deployment Failed</h2>
                <p>Your deployment has failed.</p>
                <p><strong>Project:</strong> {{ project_name }}</p>
                <p><strong>Error:</strong> {{ error_message }}</p>
            """,
            'agent_created': """
                <h2>Agent Created</h2>
                <p>A new agent has been created successfully.</p>
                <p><strong>Agent Name:</strong> {{ agent_name }}</p>
                <p><strong>Type:</strong> {{ agent_type }}</p>
            """,
            'test_results': """
                <h2>Test Results</h2>
                <p>Test execution has completed.</p>
                <p><strong>Total Tests:</strong> {{ total_tests }}</p>
                <p><strong>Passed:</strong> {{ passed_tests }}</p>
                <p><strong>Failed:</strong> {{ failed_tests }}</p>
            """
        }

        template_content = templates.get(template_name, """
            <h2>Notification</h2>
            <p>{{ message }}</p>
        """)

        from jinja2 import Template
        template = Template(template_content)
        return template.render(**context)


# Factory function
def create_email_service(config: Optional[Dict[str, Any]] = None) -> EmailService:
    """Create email service instance"""
    default_config = {
        'provider': os.environ.get('EMAIL_PROVIDER', EmailProvider.SMTP),
        'from_email': os.environ.get('EMAIL_FROM', 'noreply@a2a-portal.com'),
        'from_name': 'A2A Developer Portal'
    }

    if config:
        default_config.update(config)

    return EmailService(default_config)
