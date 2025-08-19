#!/usr/bin/env python3
"""
ORD Blockchain Adapter
Integrates existing ORD registry service with blockchain ORD registry
"""

import asyncio
import logging
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from .web3Client import get_blockchain_client

logger = logging.getLogger(__name__)

class ORDBlockchainAdapter:
    """
    Adapter that bridges existing ORD registry service with blockchain ORD registry
    """
    
    def __init__(self, traditional_ord_service=None):
        self.traditional_ord_service = traditional_ord_service
        self.blockchain_client = get_blockchain_client()
        
    async def register_ord_document(
        self,
        title: str,
        description: str,
        document_content: Dict[str, Any],
        capabilities: List[str] = None,
        tags: List[str] = None,
        dublin_core: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Register ORD document on both traditional storage and blockchain"""
        
        # Step 1: Register with traditional ORD service (for backward compatibility)
        traditional_result = None
        if self.traditional_ord_service:
            try:
                # Optional import - only if traditional ORD service is available
                try:
                    from app.ord_registry.models import RegistrationRequest
                except ImportError:
                    logger.warning("Traditional ORD registry not available")
                    return None
                
                registration_request = RegistrationRequest(
                    title=title,
                    description=description,
                    document=document_content,
                    metadata={
                        "capabilities": capabilities or [],
                        "tags": tags or [],
                        "dublin_core": dublin_core or {}
                    }
                )
                
                traditional_result = await self.traditional_ord_service.register_ord_document(
                    registration_request
                )
                logger.info(f"ORD document registered in traditional storage: {traditional_result.registration_id}")
                
            except Exception as e:
                logger.error(f"Failed to register with traditional ORD service: {e}")
        
        # Step 2: Register on blockchain
        blockchain_result = None
        try:
            # Convert document to IPFS URI or JSON string
            import json
            document_uri = f"data:application/json;base64,{json.dumps(document_content).encode().hex()}"
            
            blockchain_document_id = await self.blockchain_client.register_ord_document(
                title=title,
                description=description,
                document_uri=document_uri,
                capabilities=capabilities or [],
                tags=tags or [],
                dublin_core=dublin_core or {}
            )
            
            if blockchain_document_id:
                blockchain_result = blockchain_document_id
                logger.info(f"ORD document registered on blockchain: {blockchain_document_id}")
            
        except Exception as e:
            logger.error(f"Failed to register ORD document on blockchain: {e}")
        
        # Return combined result
        return {
            "success": traditional_result is not None or blockchain_result is not None,
            "traditional_id": traditional_result.registration_id if traditional_result else None,
            "blockchain_id": blockchain_result,
            "title": title,
            "description": description,
            "capabilities": capabilities or [],
            "tags": tags or []
        }
    
    async def search_ord_documents(
        self,
        query: str = None,
        capabilities: List[str] = None,
        tags: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Search ORD documents from both traditional storage and blockchain"""
        
        results = []
        
        # Search traditional storage
        if self.traditional_ord_service:
            try:
                # Optional import - only if traditional ORD service is available
                try:
                    from app.ord_registry.models import SearchRequest
                except ImportError:
                    logger.warning("Traditional ORD registry not available")
                    return []
                
                search_request = SearchRequest(
                    query=query or "",
                    filters={
                        "capabilities": capabilities or [],
                        "tags": tags or []
                    }
                )
                
                traditional_results = await self.traditional_ord_service.search_ord_documents(
                    search_request
                )
                
                for result in traditional_results.results:
                    results.append({
                        "source": "traditional",
                        "id": result.registration_id,
                        "title": result.title,
                        "description": result.description,
                        "score": result.score,
                        "metadata": result.metadata
                    })
                    
            except Exception as e:
                logger.error(f"Failed to search traditional ORD storage: {e}")
        
        # Search blockchain
        try:
            blockchain_results = []
            
            # Search by capabilities
            if capabilities:
                for capability in capabilities:
                    publishers = await self.blockchain_client.find_ord_documents_by_capability(capability)
                    for publisher in publishers:
                        docs = await self.blockchain_client.ord_registry_contract.functions.getDocumentsByPublisher(publisher).call()
                        for doc_id in docs:
                            doc = await self.blockchain_client.get_ord_document(doc_id.hex())
                            if doc:
                                blockchain_results.append(doc)
            
            # Search by tags
            if tags:
                for tag in tags:
                    publishers = await self.blockchain_client.find_ord_documents_by_tag(tag)
                    for publisher in publishers:
                        docs = await self.blockchain_client.ord_registry_contract.functions.getDocumentsByPublisher(publisher).call()
                        for doc_id in docs:
                            doc = await self.blockchain_client.get_ord_document(doc_id.hex())
                            if doc:
                                blockchain_results.append(doc)
            
            # Add blockchain results
            for doc in blockchain_results:
                results.append({
                    "source": "blockchain",
                    "id": doc["document_id"],
                    "title": doc["title"],
                    "description": doc["description"],
                    "publisher": doc["publisher"],
                    "document_uri": doc["document_uri"],
                    "reputation": doc["reputation"],
                    "version": doc["version"],
                    "dublin_core": doc["dublin_core"]
                })
                
        except Exception as e:
            logger.error(f"Failed to search blockchain ORD registry: {e}")
        
        # Remove duplicates and return
        unique_results = []
        seen_titles = set()
        
        for result in results:
            if result["title"] not in seen_titles:
                unique_results.append(result)
                seen_titles.add(result["title"])
        
        logger.info(f"Found {len(unique_results)} unique ORD documents ({len(results)} total)")
        return unique_results
    
    async def get_ord_document(self, document_id: str, source: str = "auto") -> Optional[Dict[str, Any]]:
        """Get ORD document by ID from specified source or auto-detect"""
        
        if source == "traditional" and self.traditional_ord_service:
            try:
                result = await self.traditional_ord_service.get_ord_registration(document_id)
                if result:
                    return {
                        "source": "traditional",
                        "id": result.registration_id,
                        "title": result.title,
                        "description": result.description,
                        "document": result.document,
                        "metadata": result.metadata,
                        "status": result.status
                    }
            except Exception as e:
                logger.error(f"Failed to get document from traditional storage: {e}")
        
        if source == "blockchain" or source == "auto":
            try:
                doc = await self.blockchain_client.get_ord_document(document_id)
                if doc:
                    return {
                        "source": "blockchain",
                        "id": doc["document_id"],
                        "title": doc["title"],
                        "description": doc["description"],
                        "publisher": doc["publisher"],
                        "document_uri": doc["document_uri"],
                        "capabilities": doc["capabilities"],
                        "tags": doc["tags"],
                        "version": doc["version"],
                        "reputation": doc["reputation"],
                        "dublin_core": doc["dublin_core"]
                    }
            except Exception as e:
                logger.error(f"Failed to get document from blockchain: {e}")
        
        return None
    
    async def enhance_ord_document(
        self,
        document_id: str,
        enhancement_type: str = "metadata_enrichment"
    ) -> Dict[str, Any]:
        """Enhance ORD document using AI and update both storages"""
        
        # Get document from traditional storage for AI enhancement
        enhanced_result = {"success": False}
        
        if self.traditional_ord_service:
            try:
                result = await self.traditional_ord_service.enhance_ord_document(
                    document_id, enhancement_type
                )
                enhanced_result = result
                logger.info(f"ORD document enhanced via traditional service: {document_id}")
                
            except Exception as e:
                logger.error(f"Failed to enhance document via traditional service: {e}")
        
        # Update blockchain version if enhancement successful
        if enhanced_result and self.blockchain_client:
            try:
                # Create transaction to update document on blockchain
                update_tx = {
                    "document_id": document_id,
                    "enhancement_type": enhancement_type,
                    "enhanced_at": datetime.now().isoformat(),
                    "enhanced_data_hash": hashlib.sha256(
                        json.dumps(enhanced_result).encode()
                    ).hexdigest()
                }
                
                # Call smart contract update method
                tx_receipt = await self.blockchain_client.call_contract_method(
                    contract_name="ORDRegistry",
                    method_name="updateORDDocument",
                    args=[
                        document_id,
                        update_tx["enhanced_data_hash"],
                        enhancement_type
                    ]
                )
                
                logger.info(f"Blockchain updated for enhanced document {document_id}: {tx_receipt}")
                
            except Exception as e:
                logger.warning(f"Failed to update blockchain for enhanced document: {e}")
                # Continue - enhancement was successful even if blockchain update failed
        
        return enhanced_result
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of both traditional and blockchain ORD systems"""
        return {
            "traditional_ord_available": self.traditional_ord_service is not None,
            "blockchain_available": self.blockchain_client is not None,
            "agent_address": self.blockchain_client.agent_identity.address if self.blockchain_client else None,
            "blockchain_balance": self.blockchain_client.get_balance() if self.blockchain_client else 0.0
        }


def create_ord_blockchain_adapter(traditional_ord_service=None) -> ORDBlockchainAdapter:
    """Factory function to create ORD blockchain adapter"""
    return ORDBlockchainAdapter(traditional_ord_service)