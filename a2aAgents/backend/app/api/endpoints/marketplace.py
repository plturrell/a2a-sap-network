"""
Marketplace API endpoints for A2A platform
Handles both data marketplace and agent service marketplace operations
"""

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import asyncio
import logging
from pydantic import BaseModel, Field

from app.services.l3DatabaseCache import L3DatabaseCache
from app.core.security import get_current_user
from app.services.recommendation_engine import RecommendationEngine, UserInteraction
from app.services.websocket_service import get_websocket_manager
from app.services.cross_marketplace_service import CrossMarketplaceService, DataConsumptionRequest
from app.services.marketplace_analytics import MarketplaceAnalytics, AnalyticsQuery, MetricType, TimeFrame
from app.models.marketplace import (
    Service, DataProduct, ServiceRequest, AgentListing,
    MarketplaceStats, RecommendationRequest, AnalyticsRequest, UserPreferences
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/marketplace", tags=["marketplace"])

# WebSocket connection manager for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = []
            self.user_connections[user_id].append(websocket)
        logger.info(f"WebSocket connected for user {user_id}")

    def disconnect(self, websocket: WebSocket, user_id: str = None):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if user_id and user_id in self.user_connections:
            if websocket in self.user_connections[user_id]:
                self.user_connections[user_id].remove(websocket)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        logger.info(f"WebSocket disconnected for user {user_id}")

    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.user_connections:
            for connection in self.user_connections[user_id]:
                try:
                    await connection.send_text(message)
                except:
                    logger.warning(f"Failed to send message to user {user_id}")

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                logger.warning("Failed to broadcast message to connection")

manager = ConnectionManager()

# Pydantic models for request/response
class CheckoutRequest(BaseModel):
    items: List[Dict[str, Any]]
    total: float
    payment_method: str = "A2A_TOKEN"

class ServiceFilter(BaseModel):
    category: Optional[str] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    rating_min: Optional[float] = None
    search_query: Optional[str] = None
    available_only: bool = True

class RecommendationResponse(BaseModel):
    services: List[Dict[str, Any]]
    data_products: List[Dict[str, Any]]
    confidence_score: float
    reasoning: str

# Initialize services
db_cache = L3DatabaseCache()
recommendation_engine = RecommendationEngine(db_cache)
cross_marketplace_service = CrossMarketplaceService(db_cache)
analytics_service = MarketplaceAnalytics(db_cache)

# Initialize WebSocket manager
websocket_manager = get_websocket_manager(db_cache)

@router.get("/services")
async def get_services(
    category: Optional[str] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    rating_min: Optional[float] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """Get available services with optional filtering"""
    try:
        # Mock data for now - replace with actual database queries
        services = [
            {
                "id": "service_001",
                "name": "AI Document Processing",
                "provider": "Agent-2",
                "description": "Advanced AI-powered document analysis and extraction",
                "category": "ai-ml",
                "price": 25.0,
                "pricing": "one-time",
                "rating": 4.8,
                "reviewCount": 127,
                "status": "Active",
                "featured": True,
                "capabilities": ["NLP", "OCR", "Data Extraction"],
                "image": "https://via.placeholder.com/300x200",
                "visible": True
            },
            {
                "id": "service_002", 
                "name": "Blockchain Analytics",
                "provider": "Agent-5",
                "description": "Real-time blockchain transaction analysis and monitoring",
                "category": "blockchain",
                "price": 50.0,
                "pricing": "subscription",
                "rating": 4.6,
                "reviewCount": 89,
                "status": "Active",
                "featured": True,
                "capabilities": ["Analytics", "Monitoring", "Alerts"],
                "image": "https://via.placeholder.com/300x200",
                "visible": True
            },
            {
                "id": "service_003",
                "name": "IoT Data Aggregation",
                "provider": "Agent-7",
                "description": "Collect and process IoT sensor data streams",
                "category": "iot",
                "price": 0.0,
                "pricing": "free",
                "rating": 4.4,
                "reviewCount": 56,
                "status": "Active",
                "featured": False,
                "capabilities": ["Data Collection", "Stream Processing", "Analytics"],
                "image": "https://via.placeholder.com/300x200",
                "visible": True
            }
        ]
        
        # Apply filters
        filtered_services = services
        if category:
            filtered_services = [s for s in filtered_services if s["category"] == category]
        if price_min is not None:
            filtered_services = [s for s in filtered_services if s["price"] >= price_min]
        if price_max is not None:
            filtered_services = [s for s in filtered_services if s["price"] <= price_max]
        if rating_min is not None:
            filtered_services = [s for s in filtered_services if s["rating"] >= rating_min]
        if search:
            search_lower = search.lower()
            filtered_services = [
                s for s in filtered_services 
                if search_lower in s["name"].lower() or 
                   search_lower in s["description"].lower() or
                   search_lower in s["provider"].lower()
            ]
        
        # Apply pagination
        total_count = len(filtered_services)
        paginated_services = filtered_services[offset:offset + limit]
        
        return {
            "services": paginated_services,
            "total": total_count,
            "offset": offset,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Error fetching services: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/data-products")
async def get_data_products(
    category: Optional[str] = None,
    format: Optional[str] = None,
    frequency: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """Get available data products with optional filtering"""
    try:
        data_products = [
            {
                "id": "data_001",
                "name": "Financial Market Data",
                "provider": "Agent-10",
                "description": "Real-time stock prices, trading volumes, and market indicators",
                "category": "financial",
                "format": "json",
                "size": "2.3 GB",
                "price": 100.0,
                "pricing": "subscription",
                "lastUpdated": datetime.now().isoformat(),
                "downloads": 1250,
                "rating": 4.9,
                "visible": True
            },
            {
                "id": "data_002",
                "name": "IoT Sensor Readings",
                "provider": "Agent-11",
                "description": "Temperature, humidity, and environmental sensor data",
                "category": "iot",
                "format": "csv",
                "size": "850 MB",
                "price": 0.0,
                "pricing": "free",
                "lastUpdated": (datetime.now() - timedelta(hours=2)).isoformat(),
                "downloads": 3450,
                "rating": 4.5,
                "visible": True
            },
            {
                "id": "data_003",
                "name": "Supply Chain Analytics",
                "provider": "Agent-12",
                "description": "Global supply chain metrics and logistics data",
                "category": "operational",
                "format": "parquet",
                "size": "1.8 GB",
                "price": 75.0,
                "pricing": "one-time",
                "lastUpdated": (datetime.now() - timedelta(days=1)).isoformat(),
                "downloads": 890,
                "rating": 4.7,
                "visible": True
            }
        ]
        
        # Apply filters
        filtered_products = data_products
        if category:
            filtered_products = [p for p in filtered_products if p["category"] == category]
        if format:
            filtered_products = [p for p in filtered_products if p["format"] == format]
        
        # Apply pagination
        total_count = len(filtered_products)
        paginated_products = filtered_products[offset:offset + limit]
        
        return {
            "data_products": paginated_products,
            "total": total_count,
            "offset": offset,
            "limit": limit
        }
        
    except Exception as e:
        logger.error(f"Error fetching data products: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/categories")
async def get_categories():
    """Get available marketplace categories"""
    try:
        categories = [
            {"id": "all", "name": "All Categories"},
            {"id": "ai-ml", "name": "AI/ML"},
            {"id": "blockchain", "name": "Blockchain"},
            {"id": "analytics", "name": "Analytics"},
            {"id": "iot", "name": "IoT"},
            {"id": "security", "name": "Security"},
            {"id": "operations", "name": "Operations"},
            {"id": "finance", "name": "Finance"},
            {"id": "financial", "name": "Financial Data"},
            {"id": "operational", "name": "Operational Data"},
            {"id": "market", "name": "Market Data"}
        ]
        return {"categories": categories}
        
    except Exception as e:
        logger.error(f"Error fetching categories: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/my-listings")
async def get_my_listings(current_user: dict = Depends(get_current_user)):
    """Get user's marketplace listings"""
    try:
        user_id = current_user.get("user_id", "default_user")
        
        # Mock data - replace with actual database queries
        listings = {
            "services": [
                {
                    "id": "user_service_001",
                    "name": "Custom Data Analysis",
                    "category": "Analytics",
                    "status": "Active",
                    "subscribers": 45,
                    "monthlyRevenue": 2250,
                    "rating": 4.7,
                    "reviews": 38
                },
                {
                    "id": "user_service_002",
                    "name": "API Integration Service",
                    "category": "Operations",
                    "status": "Paused",
                    "subscribers": 23,
                    "monthlyRevenue": 0,
                    "rating": 4.5,
                    "reviews": 15
                }
            ],
            "data": [
                {
                    "id": "user_data_001",
                    "name": "E-commerce Metrics",
                    "format": "json",
                    "size": "1.2 GB",
                    "status": "Active",
                    "downloads": 234,
                    "totalRevenue": 5670,
                    "lastUpdated": datetime.now().isoformat()
                }
            ]
        }
        
        return {"listings": listings}
        
    except Exception as e:
        logger.error(f"Error fetching user listings: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/checkout")
async def process_checkout(
    checkout_request: CheckoutRequest,
    current_user: dict = Depends(get_current_user)
):
    """Process marketplace checkout"""
    try:
        user_id = current_user.get("user_id", "default_user")
        
        # Validate items and calculate total
        calculated_total = sum(item.get("price", 0) for item in checkout_request.items)
        
        if abs(calculated_total - checkout_request.total) > 0.01:
            raise HTTPException(status_code=400, detail="Total price mismatch")
        
        # Process payment (mock implementation)
        transaction_id = f"txn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create order record
        order = {
            "transaction_id": transaction_id,
            "user_id": user_id,
            "items": checkout_request.items,
            "total": checkout_request.total,
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in database cache
        await db_cache.set_async(f"order:{transaction_id}", json.dumps(order))
        
        # Send real-time update to user
        await manager.send_personal_message(
            json.dumps({
                "type": "checkout_completed",
                "transaction_id": transaction_id,
                "total": checkout_request.total
            }),
            user_id
        )
        
        return {
            "success": True,
            "transaction_id": transaction_id,
            "message": "Checkout completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error processing checkout: {str(e)}")
        raise HTTPException(status_code=500, detail="Checkout failed")

@router.get("/services/{service_id}/trial")
@router.post("/services/{service_id}/trial")
async def launch_service_trial(service_id: str):
    """Launch service trial environment"""
    try:
        # Mock trial URL generation
        trial_url = f"https://trial.a2a.platform/service/{service_id}"
        
        return {
            "success": True,
            "trialUrl": trial_url,
            "expiresAt": (datetime.now() + timedelta(hours=24)).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error launching trial for service {service_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to launch trial")

@router.get("/data/{product_id}/preview")
async def get_data_preview(product_id: str):
    """Get data product preview"""
    try:
        # Mock preview data
        preview_data = {
            "schema": {
                "columns": ["timestamp", "value", "category", "metadata"],
                "types": ["datetime", "float", "string", "json"]
            },
            "sample_rows": [
                ["2024-01-15T10:30:00Z", 123.45, "type_a", {"source": "sensor_1"}],
                ["2024-01-15T10:31:00Z", 124.67, "type_b", {"source": "sensor_2"}],
                ["2024-01-15T10:32:00Z", 122.89, "type_a", {"source": "sensor_1"}]
            ],
            "total_rows": 1500000,
            "file_size": "2.3 GB",
            "quality_score": 0.94
        }
        
        return {"preview": preview_data}
        
    except Exception as e:
        logger.error(f"Error getting preview for data product {product_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get data preview")

@router.post("/recommendations")
async def get_recommendations(
    request: RecommendationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Get AI-powered recommendations for services and data products"""
    try:
        user_preferences = request.preferences
        context = request.context or {}
        
        # Simple recommendation engine (to be enhanced with ML)
        recommended_services = []
        recommended_data = []
        
        # Get user's past interactions and preferences
        user_id = current_user.get("user_id", "default_user")
        
        # Mock recommendation logic based on user preferences
        if "ai-ml" in user_preferences.get("categories", []):
            recommended_services.append({
                "id": "service_001",
                "name": "AI Document Processing",
                "match_score": 0.95,
                "reason": "Based on your interest in AI/ML services"
            })
        
        if "analytics" in user_preferences.get("categories", []):
            recommended_data.append({
                "id": "data_003",
                "name": "Supply Chain Analytics",
                "match_score": 0.88,
                "reason": "Matches your analytics requirements"
            })
        
        return RecommendationResponse(
            services=recommended_services,
            data_products=recommended_data,
            confidence_score=0.85,
            reasoning="Recommendations based on user preferences and interaction history"
        )
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate recommendations")

@router.get("/analytics")
async def get_marketplace_analytics(
    metric: str = "overview",
    timeframe: str = "7d",
    current_user: dict = Depends(get_current_user)
):
    """Get marketplace analytics and insights"""
    try:
        user_id = current_user.get("user_id", "default_user")
        
        analytics_data = {
            "overview": {
                "total_revenue": 125430.50,
                "total_transactions": 3421,
                "active_services": 87,
                "active_data_products": 45,
                "user_growth": 0.23,
                "revenue_growth": 0.18
            },
            "top_services": [
                {"name": "AI Document Processing", "revenue": 15600, "transactions": 234},
                {"name": "Blockchain Analytics", "revenue": 12400, "transactions": 198},
                {"name": "IoT Data Aggregation", "revenue": 8900, "transactions": 456}
            ],
            "top_data_products": [
                {"name": "Financial Market Data", "revenue": 25600, "downloads": 1250},
                {"name": "Supply Chain Analytics", "revenue": 18900, "downloads": 890},
                {"name": "IoT Sensor Readings", "revenue": 0, "downloads": 3450}
            ],
            "user_engagement": {
                "daily_active_users": 1456,
                "session_duration_avg": 24.5,
                "conversion_rate": 0.067
            },
            "geographical_distribution": [
                {"region": "North America", "percentage": 45},
                {"region": "Europe", "percentage": 32},
                {"region": "Asia Pacific", "percentage": 23}
            ]
        }
        
        return {"analytics": analytics_data, "timeframe": timeframe}
        
    except Exception as e:
        logger.error(f"Error fetching analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch analytics")

@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time marketplace updates"""
    await manager.connect(websocket, user_id)
    try:
        while True:
            # Listen for client messages
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Handle different message types
            if message_data.get("type") == "subscribe":
                # Subscribe to specific marketplace events
                subscription_types = message_data.get("subscriptions", [])
                # Store subscription preferences
                await db_cache.set_async(
                    f"websocket_subscriptions:{user_id}", 
                    json.dumps(subscription_types)
                )
                
                await websocket.send_text(json.dumps({
                    "type": "subscription_confirmed",
                    "subscriptions": subscription_types
                }))
                
            elif message_data.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
        logger.info(f"WebSocket disconnected for user {user_id}")

# Enhanced endpoints using new services

@router.post("/recommendations/enhanced")
async def get_enhanced_recommendations(
    request: RecommendationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Get AI-powered enhanced recommendations"""
    try:
        user_id = current_user.get("user_id", "default_user")
        
        # Get recommendations from the recommendation engine
        recommendations = await recommendation_engine.get_recommendations(user_id, request)
        
        # Track this interaction
        await recommendation_engine.track_interaction(UserInteraction(
            user_id=user_id,
            item_id="recommendation_request",
            item_type="interaction",
            interaction_type="view",
            timestamp=datetime.now(),
            context={"preferences": request.preferences.__dict__}
        ))
        
        return {
            "success": True,
            "recommendations": [rec.__dict__ for rec in recommendations],
            "total_count": len(recommendations),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting enhanced recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate recommendations")

@router.get("/integrations/recommendations/{agent_id}")
async def get_integration_recommendations(
    agent_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get recommendations for data products that complement an agent"""
    try:
        user_id = current_user.get("user_id", "default_user")
        recommendations = await cross_marketplace_service.get_integration_recommendations(agent_id, user_id)
        
        return {
            "success": True,
            "agent_id": agent_id,
            "recommendations": recommendations,
            "total_count": len(recommendations)
        }
        
    except Exception as e:
        logger.error(f"Error getting integration recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get integration recommendations")

@router.post("/integrations/create")
async def create_data_integration(
    request: DataConsumptionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create integration between agent service and data product"""
    try:
        result = await cross_marketplace_service.create_data_integration(request)
        return result
        
    except Exception as e:
        logger.error(f"Error creating data integration: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create integration")

@router.get("/integrations/active")
async def get_active_integrations(current_user: dict = Depends(get_current_user)):
    """Get user's active data integrations"""
    try:
        user_id = current_user.get("user_id", "default_user")
        integrations = await cross_marketplace_service.get_active_integrations(user_id)
        
        return {
            "success": True,
            "integrations": integrations,
            "total_count": len(integrations)
        }
        
    except Exception as e:
        logger.error(f"Error getting active integrations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get integrations")

@router.get("/integrations/{pipeline_id}/analytics")
async def get_integration_analytics(
    pipeline_id: str,
    timeframe: str = "7d",
    current_user: dict = Depends(get_current_user)
):
    """Get analytics for a specific data integration"""
    try:
        analytics = await cross_marketplace_service.get_integration_analytics(pipeline_id, timeframe)
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting integration analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get integration analytics")

@router.post("/integrations/{pipeline_id}/pause")
async def pause_integration(
    pipeline_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Pause a data integration"""
    try:
        user_id = current_user.get("user_id", "default_user")
        result = await cross_marketplace_service.pause_integration(pipeline_id, user_id)
        return result
        
    except Exception as e:
        logger.error(f"Error pausing integration: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to pause integration")

@router.post("/integrations/{pipeline_id}/resume")
async def resume_integration(
    pipeline_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Resume a paused data integration"""
    try:
        user_id = current_user.get("user_id", "default_user")
        result = await cross_marketplace_service.resume_integration(pipeline_id, user_id)
        return result
        
    except Exception as e:
        logger.error(f"Error resuming integration: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to resume integration")

@router.delete("/integrations/{pipeline_id}")
async def delete_integration(
    pipeline_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a data integration"""
    try:
        user_id = current_user.get("user_id", "default_user")
        result = await cross_marketplace_service.delete_integration(pipeline_id, user_id)
        return result
        
    except Exception as e:
        logger.error(f"Error deleting integration: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete integration")

@router.get("/analytics/overview")
async def get_analytics_overview(
    timeframe: str = "7d",
    current_user: dict = Depends(get_current_user)
):
    """Get marketplace analytics overview"""
    try:
        overview = await analytics_service.get_marketplace_overview(timeframe)
        return overview
        
    except Exception as e:
        logger.error(f"Error getting analytics overview: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get analytics overview")

@router.get("/analytics/revenue")
async def get_revenue_analytics(
    timeframe: str = "30d",
    breakdown: str = "daily",
    current_user: dict = Depends(get_current_user)
):
    """Get detailed revenue analytics"""
    try:
        revenue_data = await analytics_service.get_revenue_analytics(timeframe, breakdown)
        return revenue_data
        
    except Exception as e:
        logger.error(f"Error getting revenue analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get revenue analytics")

@router.get("/analytics/engagement")
async def get_engagement_analytics(
    timeframe: str = "30d",
    current_user: dict = Depends(get_current_user)
):
    """Get user engagement analytics"""
    try:
        engagement_data = await analytics_service.get_user_engagement_metrics(timeframe)
        return engagement_data
        
    except Exception as e:
        logger.error(f"Error getting engagement analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get engagement analytics")

@router.get("/analytics/performance")
async def get_performance_analytics(
    timeframe: str = "30d",
    current_user: dict = Depends(get_current_user)
):
    """Get service and agent performance analytics"""
    try:
        performance_data = await analytics_service.get_service_performance_analytics(timeframe)
        return performance_data
        
    except Exception as e:
        logger.error(f"Error getting performance analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get performance analytics")

@router.get("/analytics/data-products")
async def get_data_product_analytics(
    timeframe: str = "30d",
    current_user: dict = Depends(get_current_user)
):
    """Get data product analytics"""
    try:
        data_analytics = await analytics_service.get_data_product_analytics(timeframe)
        return data_analytics
        
    except Exception as e:
        logger.error(f"Error getting data product analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get data product analytics")

@router.get("/analytics/predictive")
async def get_predictive_analytics(
    forecast_days: int = 30,
    current_user: dict = Depends(get_current_user)
):
    """Get predictive analytics and forecasts"""
    try:
        predictions = await analytics_service.get_predictive_analytics(forecast_days)
        return predictions
        
    except Exception as e:
        logger.error(f"Error getting predictive analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get predictive analytics")

@router.get("/analytics/competitive")
async def get_competitive_analysis(current_user: dict = Depends(get_current_user)):
    """Get competitive landscape analysis"""
    try:
        competitive_data = await analytics_service.get_competitive_analysis()
        return competitive_data
        
    except Exception as e:
        logger.error(f"Error getting competitive analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get competitive analysis")

@router.post("/analytics/custom-report")
async def generate_custom_analytics_report(
    query: AnalyticsQuery,
    current_user: dict = Depends(get_current_user)
):
    """Generate custom analytics report"""
    try:
        result = await analytics_service.generate_custom_report(query)
        return {
            "success": True,
            "query_id": result.query_id,
            "data": result.data,
            "metadata": result.metadata,
            "generated_at": result.generated_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating custom report: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate custom report")

@router.post("/interactions/track")
async def track_user_interaction(
    interaction: UserInteraction,
    current_user: dict = Depends(get_current_user)
):
    """Track user interaction for improving recommendations"""
    try:
        await recommendation_engine.track_interaction(interaction)
        
        return {
            "success": True,
            "message": "Interaction tracked successfully"
        }
        
    except Exception as e:
        logger.error(f"Error tracking interaction: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to track interaction")

# Background task for sending periodic updates
async def broadcast_marketplace_updates():
    """Background task to broadcast marketplace updates"""
    while True:
        try:
            # Get latest marketplace stats from analytics service
            overview = await analytics_service.get_marketplace_overview("1h")
            
            update_data = {
                "type": "marketplace_update",
                "timestamp": datetime.now().isoformat(),
                "stats": overview.get("health_metrics", {})
            }
            
            if websocket_manager:
                await websocket_manager.broadcast_marketplace_update(update_data)
            
        except Exception as e:
            logger.error(f"Error broadcasting marketplace updates: {str(e)}")
        
        # Wait 30 seconds before next broadcast
        await asyncio.sleep(30)

# Start background task
asyncio.create_task(broadcast_marketplace_updates())