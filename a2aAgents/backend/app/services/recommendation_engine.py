"""
Advanced AI-powered recommendation engine for A2A marketplace
Provides intelligent matching of services and data products based on user preferences and behavior
"""

import asyncio
import logging
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pandas as pd

from app.services.l3DatabaseCache import L3DatabaseCache
from app.models.marketplace import (
    Service, DataProduct, AgentListing, UserPreferences, 
    RecommendationRequest, RecommendationResult
)

logger = logging.getLogger(__name__)

@dataclass
class UserInteraction:
    user_id: str
    item_id: str
    item_type: str  # "service", "data_product", "agent"
    interaction_type: str  # "view", "purchase", "request", "rate"
    timestamp: datetime
    rating: Optional[float] = None
    context: Optional[Dict[str, Any]] = None

@dataclass
class ContentFeatures:
    text_features: np.ndarray
    categorical_features: Dict[str, Any]
    numerical_features: Dict[str, float]

class RecommendationEngine:
    """
    Hybrid recommendation engine combining multiple approaches:
    1. Content-based filtering
    2. Collaborative filtering
    3. Context-aware recommendations
    4. Cross-marketplace recommendations
    """
    
    def __init__(self, db_cache: L3DatabaseCache):
        self.db_cache = db_cache
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.user_interactions: Dict[str, List[UserInteraction]] = {}
        self.item_features: Dict[str, ContentFeatures] = {}
        self.user_clusters: Dict[str, int] = {}
        
        # Initialize with mock data
        asyncio.create_task(self._initialize_engine())
    
    async def _initialize_engine(self):
        """Initialize the recommendation engine with existing data"""
        try:
            # Load existing interactions from cache
            interactions_data = await self.db_cache.get_async("user_interactions:all")
            if interactions_data:
                self._load_interactions_from_data(json.loads(interactions_data))
            
            # Load item features
            await self._build_item_features()
            
            # Perform user clustering
            await self._cluster_users()
            
            logger.info("Recommendation engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing recommendation engine: {str(e)}")
    
    async def get_recommendations(
        self, 
        user_id: str,
        request: RecommendationRequest
    ) -> List[RecommendationResult]:
        """Get personalized recommendations for a user"""
        try:
            # Get user's interaction history
            user_history = self.user_interactions.get(user_id, [])
            
            # Generate recommendations using multiple strategies
            content_recs = await self._content_based_recommendations(user_id, request)
            collaborative_recs = await self._collaborative_filtering(user_id, request)
            context_recs = await self._context_aware_recommendations(user_id, request)
            cross_market_recs = await self._cross_marketplace_recommendations(user_id, request)
            
            # Combine and rank recommendations
            all_recommendations = self._combine_recommendations([
                content_recs,
                collaborative_recs, 
                context_recs,
                cross_market_recs
            ])
            
            # Apply diversity and novelty filtering
            final_recs = self._apply_diversity_filter(all_recommendations, request.limit)
            
            # Log recommendation event
            await self._log_recommendation_event(user_id, final_recs)
            
            return final_recs
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {str(e)}")
            return await self._fallback_recommendations(request)
    
    async def _content_based_recommendations(
        self, 
        user_id: str,
        request: RecommendationRequest
    ) -> List[RecommendationResult]:
        """Content-based filtering using item features"""
        recommendations = []
        
        try:
            # Get user preferences
            preferences = request.preferences
            
            # Create user profile from preferences and history
            user_profile = await self._create_user_profile(user_id, preferences)
            
            # Get all available items
            services = await self._get_available_services()
            data_products = await self._get_available_data_products()
            
            # Calculate similarity scores
            for service in services:
                if self._meets_basic_criteria(service, preferences):
                    score = self._calculate_content_similarity(user_profile, service)
                    if score > 0.3:  # Threshold for relevance
                        recommendations.append(RecommendationResult(
                            item_id=service["id"],
                            item_type="service",
                            match_score=score,
                            reason=f"Matches your interest in {service['category']} services"
                        ))
            
            for product in data_products:
                if self._meets_basic_criteria(product, preferences):
                    score = self._calculate_content_similarity(user_profile, product)
                    if score > 0.3:
                        recommendations.append(RecommendationResult(
                            item_id=product["id"],
                            item_type="data_product", 
                            match_score=score,
                            reason=f"High-quality {product['category']} data matching your needs"
                        ))
            
            # Sort by score
            recommendations.sort(key=lambda x: x.match_score, reverse=True)
            return recommendations[:request.limit // 2]
            
        except Exception as e:
            logger.error(f"Error in content-based recommendations: {str(e)}")
            return []
    
    async def _collaborative_filtering(
        self,
        user_id: str,
        request: RecommendationRequest
    ) -> List[RecommendationResult]:
        """Collaborative filtering using user similarity"""
        recommendations = []
        
        try:
            # Find similar users
            similar_users = await self._find_similar_users(user_id)
            
            # Get items liked by similar users
            user_history = self.user_interactions.get(user_id, [])
            user_items = {interaction.item_id for interaction in user_history}
            
            item_scores = {}
            
            for similar_user_id, similarity_score in similar_users[:10]:  # Top 10 similar users
                similar_user_history = self.user_interactions.get(similar_user_id, [])
                
                for interaction in similar_user_history:
                    if (interaction.item_id not in user_items and 
                        interaction.interaction_type in ["purchase", "rate"] and
                        (not interaction.rating or interaction.rating >= 4.0)):
                        
                        if interaction.item_id not in item_scores:
                            item_scores[interaction.item_id] = 0
                        
                        # Weight by user similarity and interaction strength
                        weight = similarity_score * (interaction.rating or 4.0) / 5.0
                        item_scores[interaction.item_id] += weight
            
            # Convert to recommendations
            for item_id, score in sorted(item_scores.items(), key=lambda x: x[1], reverse=True):
                item_type = await self._get_item_type(item_id)
                if item_type:
                    recommendations.append(RecommendationResult(
                        item_id=item_id,
                        item_type=item_type,
                        match_score=min(score, 1.0),
                        reason="Recommended by users with similar preferences"
                    ))
            
            return recommendations[:request.limit // 4]
            
        except Exception as e:
            logger.error(f"Error in collaborative filtering: {str(e)}")
            return []
    
    async def _context_aware_recommendations(
        self,
        user_id: str,
        request: RecommendationRequest
    ) -> List[RecommendationResult]:
        """Context-aware recommendations based on current situation"""
        recommendations = []
        
        try:
            context = request.context or {}
            current_time = datetime.now()
            
            # Time-based recommendations
            if current_time.hour < 12:  # Morning
                recommendations.extend(await self._get_morning_recommendations(user_id, request))
            elif current_time.hour > 17:  # Evening
                recommendations.extend(await self._get_evening_recommendations(user_id, request))
            
            # Project-based recommendations
            if context.get("current_project"):
                project_type = context["current_project"].get("type")
                recommendations.extend(await self._get_project_recommendations(user_id, project_type))
            
            # Seasonal recommendations
            month = current_time.month
            if month in [12, 1, 2]:  # Winter
                recommendations.extend(await self._get_seasonal_recommendations(user_id, "winter"))
            elif month in [6, 7, 8]:  # Summer
                recommendations.extend(await self._get_seasonal_recommendations(user_id, "summer"))
            
            return recommendations[:request.limit // 4]
            
        except Exception as e:
            logger.error(f"Error in context-aware recommendations: {str(e)}")
            return []
    
    async def _cross_marketplace_recommendations(
        self,
        user_id: str,
        request: RecommendationRequest
    ) -> List[RecommendationResult]:
        """Cross-marketplace recommendations (agents + data products)"""
        recommendations = []
        
        try:
            # If user frequently uses AI services, recommend relevant data products
            user_history = self.user_interactions.get(user_id, [])
            service_interactions = [i for i in user_history if i.item_type == "service"]
            
            # Analyze service usage patterns
            service_categories = {}
            for interaction in service_interactions:
                # Mock category extraction - in real implementation, fetch from DB
                category = await self._get_item_category(interaction.item_id)
                if category:
                    service_categories[category] = service_categories.get(category, 0) + 1
            
            # Recommend complementary data products
            if "ai-ml" in service_categories:
                ai_data_products = await self._get_data_products_by_category("training-data")
                for product in ai_data_products[:3]:
                    recommendations.append(RecommendationResult(
                        item_id=product["id"],
                        item_type="data_product",
                        match_score=0.8,
                        reason="High-quality training data for your AI services"
                    ))
            
            if "analytics" in service_categories:
                analytics_data = await self._get_data_products_by_category("business-intelligence")
                for product in analytics_data[:2]:
                    recommendations.append(RecommendationResult(
                        item_id=product["id"],
                        item_type="data_product",
                        match_score=0.75,
                        reason="Business data to enhance your analytics workflows"
                    ))
            
            # Recommend agents based on data product usage
            data_interactions = [i for i in user_history if i.item_type == "data_product"]
            if data_interactions:
                # Suggest processing agents for data products
                processing_agents = await self._get_agents_by_capability("data-processing")
                for agent in processing_agents[:2]:
                    recommendations.append(RecommendationResult(
                        item_id=agent["id"],
                        item_type="service",
                        match_score=0.7,
                        reason="Data processing services for your datasets"
                    ))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in cross-marketplace recommendations: {str(e)}")
            return []
    
    def _combine_recommendations(self, rec_lists: List[List[RecommendationResult]]) -> List[RecommendationResult]:
        """Combine recommendations from multiple strategies"""
        combined = {}
        
        for rec_list in rec_lists:
            for rec in rec_list:
                key = f"{rec.item_type}:{rec.item_id}"
                if key in combined:
                    # Boost score for items recommended by multiple strategies
                    combined[key].match_score = min(
                        combined[key].match_score + rec.match_score * 0.3, 
                        1.0
                    )
                else:
                    combined[key] = rec
        
        return list(combined.values())
    
    def _apply_diversity_filter(
        self, 
        recommendations: List[RecommendationResult], 
        limit: int
    ) -> List[RecommendationResult]:
        """Apply diversity filtering to avoid over-concentration in one category"""
        if not recommendations:
            return []
        
        # Sort by score first
        recommendations.sort(key=lambda x: x.match_score, reverse=True)
        
        # Apply diversity constraints
        selected = []
        category_counts = {}
        type_counts = {"service": 0, "data_product": 0}
        
        for rec in recommendations:
            if len(selected) >= limit:
                break
            
            # Get category (mock implementation)
            category = "general"  # In real implementation, fetch from item data
            
            # Ensure diversity
            if (category_counts.get(category, 0) < limit // 3 and 
                type_counts[rec.item_type] < limit // 2):
                
                selected.append(rec)
                category_counts[category] = category_counts.get(category, 0) + 1
                type_counts[rec.item_type] += 1
        
        return selected
    
    async def track_interaction(self, interaction: UserInteraction):
        """Track user interaction for improving recommendations"""
        try:
            user_id = interaction.user_id
            
            if user_id not in self.user_interactions:
                self.user_interactions[user_id] = []
            
            self.user_interactions[user_id].append(interaction)
            
            # Keep only recent interactions (last 1000 per user)
            if len(self.user_interactions[user_id]) > 1000:
                self.user_interactions[user_id] = self.user_interactions[user_id][-1000:]
            
            # Persist to cache
            await self.db_cache.set_async(
                f"user_interactions:{user_id}",
                json.dumps([{
                    "item_id": i.item_id,
                    "item_type": i.item_type,
                    "interaction_type": i.interaction_type,
                    "timestamp": i.timestamp.isoformat(),
                    "rating": i.rating,
                    "context": i.context
                } for i in self.user_interactions[user_id][-100:]])  # Store last 100
            )
            
            # Update user clusters periodically
            if len(self.user_interactions[user_id]) % 10 == 0:  # Every 10 interactions
                await self._update_user_cluster(user_id)
            
            logger.debug(f"Tracked interaction: {interaction.interaction_type} for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error tracking interaction: {str(e)}")
    
    # Helper methods (simplified implementations)
    
    async def _get_available_services(self) -> List[Dict[str, Any]]:
        """Get all available services - mock implementation"""
        return [
            {
                "id": "service_001",
                "name": "AI Document Processing",
                "category": "ai-ml",
                "description": "Advanced AI-powered document analysis",
                "price": 25.0,
                "rating": 4.8,
                "capabilities": ["NLP", "OCR", "Data Extraction"]
            },
            {
                "id": "service_002",
                "name": "Blockchain Analytics",
                "category": "blockchain", 
                "description": "Real-time blockchain analysis",
                "price": 50.0,
                "rating": 4.6,
                "capabilities": ["Analytics", "Monitoring", "Alerts"]
            }
        ]
    
    async def _get_available_data_products(self) -> List[Dict[str, Any]]:
        """Get all available data products - mock implementation"""
        return [
            {
                "id": "data_001",
                "name": "Financial Market Data",
                "category": "financial",
                "description": "Real-time market data",
                "price": 100.0,
                "quality_score": 0.95,
                "format": "json"
            },
            {
                "id": "data_002",
                "name": "IoT Sensor Data",
                "category": "iot",
                "description": "Environmental sensor readings", 
                "price": 0.0,
                "quality_score": 0.88,
                "format": "csv"
            }
        ]
    
    def _meets_basic_criteria(self, item: Dict[str, Any], preferences: UserPreferences) -> bool:
        """Check if item meets user's basic criteria"""
        # Price range check
        price_range = preferences.price_range
        if not (price_range["min"] <= item.get("price", 0) <= price_range["max"]):
            return False
        
        # Rating threshold check
        if item.get("rating", 0) < preferences.rating_threshold:
            return False
        
        # Category preference check
        if preferences.categories and item.get("category") not in preferences.categories:
            return False
        
        return True
    
    def _calculate_content_similarity(self, user_profile: Dict, item: Dict[str, Any]) -> float:
        """Calculate content similarity between user profile and item"""
        # Simple similarity calculation based on category and description matching
        score = 0.0
        
        # Category match
        if item.get("category") in user_profile.get("preferred_categories", []):
            score += 0.4
        
        # Description similarity (simplified)
        user_interests = user_profile.get("interests", [])
        item_desc = item.get("description", "").lower()
        
        matching_interests = sum(1 for interest in user_interests if interest.lower() in item_desc)
        if user_interests:
            score += 0.3 * (matching_interests / len(user_interests))
        
        # Price preference
        preferred_price = user_profile.get("preferred_price_range", {})
        item_price = item.get("price", 0)
        if preferred_price.get("min", 0) <= item_price <= preferred_price.get("max", 1000):
            score += 0.2
        
        # Quality/rating boost
        if item.get("rating", 0) >= 4.5:
            score += 0.1
        
        return min(score, 1.0)
    
    async def _create_user_profile(self, user_id: str, preferences: UserPreferences) -> Dict[str, Any]:
        """Create user profile from preferences and history"""
        profile = {
            "preferred_categories": preferences.categories,
            "preferred_price_range": preferences.price_range,
            "preferred_providers": preferences.preferred_providers,
            "interests": []
        }
        
        # Extract interests from interaction history
        user_history = self.user_interactions.get(user_id, [])
        for interaction in user_history[-50:]:  # Last 50 interactions
            if interaction.rating and interaction.rating >= 4.0:
                # Add positive interaction context as interests
                if interaction.context and "keywords" in interaction.context:
                    profile["interests"].extend(interaction.context["keywords"])
        
        return profile
    
    async def _find_similar_users(self, user_id: str) -> List[Tuple[str, float]]:
        """Find users with similar preferences and behavior"""
        similar_users = []
        
        # Get user's interaction pattern
        user_history = self.user_interactions.get(user_id, [])
        if not user_history:
            return []
        
        user_items = {interaction.item_id for interaction in user_history}
        
        # Compare with other users
        for other_user_id, other_history in self.user_interactions.items():
            if other_user_id == user_id or not other_history:
                continue
            
            other_items = {interaction.item_id for interaction in other_history}
            
            # Calculate Jaccard similarity
            intersection = len(user_items & other_items)
            union = len(user_items | other_items)
            
            if union > 0:
                similarity = intersection / union
                if similarity > 0.1:  # Minimum similarity threshold
                    similar_users.append((other_user_id, similarity))
        
        # Sort by similarity
        similar_users.sort(key=lambda x: x[1], reverse=True)
        return similar_users
    
    async def _fallback_recommendations(self, request: RecommendationRequest) -> List[RecommendationResult]:
        """Fallback recommendations when other methods fail"""
        fallback_recs = []
        
        # Popular items from each category
        popular_services = [
            ("service_001", "Popular AI service with high ratings"),
            ("service_002", "Trending blockchain analytics tool")
        ]
        
        popular_data = [
            ("data_001", "High-quality financial dataset"),
            ("data_002", "Comprehensive IoT sensor data")
        ]
        
        for item_id, reason in popular_services[:request.limit // 2]:
            fallback_recs.append(RecommendationResult(
                item_id=item_id,
                item_type="service",
                match_score=0.6,
                reason=reason
            ))
        
        for item_id, reason in popular_data[:request.limit // 2]:
            fallback_recs.append(RecommendationResult(
                item_id=item_id,
                item_type="data_product", 
                match_score=0.6,
                reason=reason
            ))
        
        return fallback_recs
    
    async def _log_recommendation_event(self, user_id: str, recommendations: List[RecommendationResult]):
        """Log recommendation event for analytics"""
        event_data = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "recommendations": [
                {
                    "item_id": rec.item_id,
                    "item_type": rec.item_type,
                    "match_score": rec.match_score,
                    "reason": rec.reason
                }
                for rec in recommendations
            ]
        }
        
        await self.db_cache.set_async(
            f"recommendation_event:{user_id}:{datetime.now().timestamp()}",
            json.dumps(event_data)
        )
    
    # Placeholder methods for full implementation
    async def _build_item_features(self):
        """Build feature vectors for all items"""
        pass
    
    async def _cluster_users(self):
        """Cluster users based on behavior patterns"""
        pass
    
    async def _update_user_cluster(self, user_id: str):
        """Update user's cluster assignment"""
        pass
    
    async def _get_item_type(self, item_id: str) -> Optional[str]:
        """Get item type from item ID"""
        if item_id.startswith("service_"):
            return "service"
        elif item_id.startswith("data_"):
            return "data_product"
        return None
    
    async def _get_item_category(self, item_id: str) -> Optional[str]:
        """Get item category from item ID"""
        # Mock implementation
        category_mapping = {
            "service_001": "ai-ml",
            "service_002": "blockchain",
            "data_001": "financial",
            "data_002": "iot"
        }
        return category_mapping.get(item_id)
    
    async def _get_data_products_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get data products by category"""
        # Mock implementation
        return await self._get_available_data_products()
    
    async def _get_agents_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """Get agents by capability"""
        # Mock implementation
        return await self._get_available_services()
    
    async def _get_morning_recommendations(self, user_id: str, request: RecommendationRequest) -> List[RecommendationResult]:
        """Get morning-specific recommendations"""
        return []
    
    async def _get_evening_recommendations(self, user_id: str, request: RecommendationRequest) -> List[RecommendationResult]:
        """Get evening-specific recommendations"""
        return []
    
    async def _get_project_recommendations(self, user_id: str, project_type: str) -> List[RecommendationResult]:
        """Get project-specific recommendations"""
        return []
    
    async def _get_seasonal_recommendations(self, user_id: str, season: str) -> List[RecommendationResult]:
        """Get seasonal recommendations"""
        return []
    
    def _load_interactions_from_data(self, data: Dict[str, List[Dict]]):
        """Load user interactions from stored data"""
        for user_id, interactions_data in data.items():
            self.user_interactions[user_id] = [
                UserInteraction(
                    user_id=user_id,
                    item_id=i["item_id"],
                    item_type=i["item_type"],
                    interaction_type=i["interaction_type"],
                    timestamp=datetime.fromisoformat(i["timestamp"]),
                    rating=i.get("rating"),
                    context=i.get("context")
                )
                for i in interactions_data
            ]