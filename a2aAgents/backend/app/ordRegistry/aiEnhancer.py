"""
AI Enhancement Module for ORD Registry
Using Grok and Perplexity A2A Clients for Intelligent Metadata Generation
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from .models import ORDDocument, DublinCoreMetadata
from ..clients.grokClient import GrokClient
from ..clients.perplexityClient import PerplexityClient

logger = logging.getLogger(__name__)


class ORDAIEnhancer:
    """AI-powered enhancement for ORD documents using A2A clients"""

    def __init__(self, grok_client: GrokClient = None, perplexity_client: PerplexityClient = None):
        self.grok_client = grok_client
        self.perplexity_client = perplexity_client

    async def enhance_ord_document(self, ord_document: ORDDocument) -> ORDDocument:
        """Enhance ORD document with AI-generated Dublin Core metadata and descriptions"""
        try:
            # Create a copy to avoid modifying the original
            enhanced_doc = ord_document.copy(deep=True)

            # Generate enhanced Dublin Core metadata if missing or incomplete
            if not enhanced_doc.dublinCore or self._needs_dublin_core_enhancement(enhanced_doc.dublinCore):
                enhanced_dublin_core = await self._generate_dublin_core_metadata(enhanced_doc)
                enhanced_doc.dublinCore = enhanced_dublin_core

            # Enhance resource descriptions
            enhanced_doc = await self._enhance_resource_descriptions(enhanced_doc)

            # Enrich with industry-standard tags and classifications
            enhanced_doc = await self._enrich_with_standards(enhanced_doc)

            logger.info("✅ ORD document enhanced with AI-generated metadata")
            return enhanced_doc

        except Exception as e:
            logger.warning(f"AI enhancement failed, returning original document: {e}")
            return ord_document

    def _needs_dublin_core_enhancement(self, dublin_core: DublinCoreMetadata) -> bool:
        """Check if Dublin Core metadata needs enhancement"""
        if not dublin_core:
            return True

        # Check completion of core elements
        core_elements = [
            dublin_core.title, dublin_core.description, dublin_core.creator,
            dublin_core.subject, dublin_core.publisher, dublin_core.type
        ]

        missing_count = sum(1 for element in core_elements if not element)
        return missing_count > 2  # Enhance if more than 2 core elements are missing

    async def _generate_dublin_core_metadata(self, ord_document: ORDDocument) -> DublinCoreMetadata:
        """Generate comprehensive Dublin Core metadata using Grok"""
        try:
            if not self.grok_client:
                logger.warning("Grok client not available for Dublin Core generation")
                return ord_document.dublinCore or DublinCoreMetadata()

            # Extract content for analysis
            content_summary = self._extract_document_content(ord_document)

            # Generate Dublin Core metadata using Grok
            dublin_core_prompt = f"""
            Analyze this ORD document and generate comprehensive Dublin Core metadata:

            Document Content:
            {content_summary}

            Generate Dublin Core metadata with these elements:
            - title: Clear, descriptive title
            - creator: Organization or person responsible
            - subject: Key topics and themes (as array)
            - description: Comprehensive description
            - publisher: Publishing organization
            - contributor: Additional contributors (as array)
            - date: Current date in ISO format
            - type: Resource type (e.g., "Dataset", "API", "Service")
            - format: Technical format information
            - identifier: Unique identifier
            - language: Primary language (e.g., "en")
            - coverage: Scope or domain coverage
            - rights: Usage rights and licensing

            Return ONLY valid JSON with these Dublin Core elements.
            """

            response = await self.grok_client.async_chat_completion(
                messages=[{"role": "user", "content": dublin_core_prompt}],
                max_tokens=500,
                temperature=0.3
            )

            # Parse Grok response and create Dublin Core metadata
            dublin_core_data = self._parse_grok_dublin_core(response.content)
            return DublinCoreMetadata(**dublin_core_data)

        except Exception as e:
            logger.error(f"Dublin Core generation failed: {e}")
            return ord_document.dublinCore or DublinCoreMetadata()

    def _extract_document_content(self, ord_document: ORDDocument) -> str:
        """Extract key content from ORD document for AI analysis"""
        content_parts = []

        # Add basic document info
        if ord_document.description:
            content_parts.append(f"Description: {ord_document.description}")

        # Add resource information
        for resource_type in ["dataProducts", "apiResources", "entityTypes", "eventResources"]:
            resources = getattr(ord_document, resource_type, None)
            if resources:
                content_parts.append(f"\n{resource_type.upper()}:")
                for resource in resources[:3]:  # Limit to first 3 for analysis
                    if isinstance(resource, dict):
                        title = resource.get("title", "Unknown")
                        desc = resource.get("description", resource.get("shortDescription", ""))
                        content_parts.append(f"- {title}: {desc}")

        return "\n".join(content_parts)[:2000]  # Limit length for AI processing

    def _parse_grok_dublin_core(self, grok_response: str) -> Dict[str, Any]:
        """Parse Grok response into Dublin Core data"""
        try:
            import json
            # Try to extract JSON from response
            if '{' in grok_response and '}' in grok_response:
                json_start = grok_response.find('{')
                json_end = grok_response.rfind('}') + 1
                json_str = grok_response[json_start:json_end]
                return json.loads(json_str)
            else:
                # Fallback to basic metadata
                return self._generate_fallback_dublin_core(grok_response)

        except Exception as e:
            logger.warning(f"Failed to parse Grok Dublin Core response: {e}")
            return self._generate_fallback_dublin_core(grok_response)

    def _generate_fallback_dublin_core(self, content: str) -> Dict[str, Any]:
        """Generate basic Dublin Core metadata as fallback"""
        return {
            "title": "Enterprise Resource",
            "description": content[:200] if content else "Auto-generated resource description",
            "creator": ["AI-Enhanced ORD Registry"],
            "subject": ["enterprise", "data", "api"],
            "publisher": "FinSight CIB A2A System",
            "date": datetime.utcnow().isoformat(),
            "type": "Resource",
            "language": "en",
            "rights": "Internal use - enterprise data governance applies"
        }

    async def _enhance_resource_descriptions(self, ord_document: ORDDocument) -> ORDDocument:
        """Enhance resource descriptions using AI"""
        try:
            if not self.grok_client:
                return ord_document

            # Enhance data products
            if ord_document.dataProducts:
                for resource in ord_document.dataProducts:
                    if isinstance(resource, dict) and not resource.get("description"):
                        enhanced_desc = await self._generate_resource_description(
                            resource.get("title", ""),
                            resource.get("shortDescription", ""),
                            "data product"
                        )
                        resource["description"] = enhanced_desc

            # Enhance API resources
            if ord_document.apiResources:
                for resource in ord_document.apiResources:
                    if isinstance(resource, dict) and not resource.get("description"):
                        enhanced_desc = await self._generate_resource_description(
                            resource.get("title", ""),
                            resource.get("shortDescription", ""),
                            "API"
                        )
                        resource["description"] = enhanced_desc

            return ord_document

        except Exception as e:
            logger.warning(f"Resource description enhancement failed: {e}")
            return ord_document

    async def _generate_resource_description(self, title: str, short_desc: str, resource_type: str) -> str:
        """Generate enhanced description for a resource"""
        try:
            description_prompt = f"""
            Generate a comprehensive, professional description for this {resource_type}:

            Title: {title}
            Short Description: {short_desc}
            Resource Type: {resource_type}

            Create a detailed, technical description (2-3 sentences) that:
            1. Explains the purpose and functionality
            2. Describes key features or capabilities
            3. Mentions typical use cases or integration points

            Make it professional and suitable for enterprise documentation.
            Return only the description text, no formatting.
            """

            response = await self.grok_client.async_chat_completion(
                messages=[{"role": "user", "content": description_prompt}],
                max_tokens=150,
                temperature=0.2
            )

            return response.content.strip() if response.content else short_desc

        except Exception as e:
            logger.warning(f"Description generation failed: {e}")
            return short_desc or f"Enterprise {resource_type} resource"

    async def _enrich_with_standards(self, ord_document: ORDDocument) -> ORDDocument:
        """Enrich document with industry-standard classifications using Perplexity"""
        try:
            if not self.perplexity_client:
                return ord_document

            # Extract domain information
            content_summary = self._extract_document_content(ord_document)

            # Query Perplexity for industry standards and classifications
            standards_query = f"""
            For this enterprise resource, suggest relevant industry standards,
            classifications, and metadata tags based on current best practices:

            {content_summary[:500]}

            Focus on: ISO standards, industry classifications, common metadata tags
            """

            standards_response = await self.perplexity_client.search_real_time(
                query=standards_query,
                max_tokens=200
            )

            # Parse and apply standards (simplified for this implementation)
            if standards_response and standards_response.content:
                # Add industry-relevant tags
                if ord_document.dublinCore:
                    if not ord_document.dublinCore.subject:
                        ord_document.dublinCore.subject = []

                    # Add standard tags based on content
                    standard_tags = self._extract_standard_tags(standards_response.content)
                    ord_document.dublinCore.subject.extend(standard_tags)

                    # Remove duplicates
                    ord_document.dublinCore.subject = list(set(ord_document.dublinCore.subject))

            return ord_document

        except Exception as e:
            logger.warning(f"Standards enrichment failed: {e}")
            return ord_document

    def _extract_standard_tags(self, standards_content: str) -> List[str]:
        """Extract relevant standard tags from Perplexity response"""
        # Simplified tag extraction - in production, use more sophisticated NLP
        common_standards = [
            "iso27001", "gdpr", "financial-services", "data-governance",
            "api-management", "enterprise-data", "compliance", "metadata-management"
        ]

        content_lower = standards_content.lower()
        relevant_tags = [tag for tag in common_standards if tag.replace("-", " ") in content_lower]

        return relevant_tags[:5]  # Limit to 5 most relevant tags


# Factory function for creating AI enhancer
def create_ord_ai_enhancer(grok_client: GrokClient = None, perplexity_client: PerplexityClient = None) -> ORDAIEnhancer:
    """Create an ORD AI enhancer with the provided clients"""
    return ORDAIEnhancer(grok_client=grok_client, perplexity_client=perplexity_client)
