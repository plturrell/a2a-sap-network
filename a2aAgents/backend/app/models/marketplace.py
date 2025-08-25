"""
Pydantic models for marketplace operations
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

class ServiceStatus(str, Enum):
    LISTED = "listed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DISPUTED = "disputed"
    CANCELLED = "cancelled"

class ServiceType(str, Enum):
    ONE_TIME = "one-time"
    SUBSCRIPTION = "subscription"
    ON_DEMAND = "on-demand"

class PricingModel(str, Enum):
    FREE = "free"
    ONE_TIME = "one-time"
    SUBSCRIPTION = "subscription"
    PAY_PER_USE = "pay-per-use"

class Service(BaseModel):
    id: str
    name: str
    provider: str
    description: str
    category: str
    price: float
    pricing: PricingModel
    rating: float = Field(ge=0, le=5)
    review_count: int = Field(ge=0)
    status: str
    featured: bool = False
    capabilities: List[str]
    image: Optional[str] = None
    visible: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class DataProduct(BaseModel):
    id: str
    name: str
    provider: str
    description: str
    category: str
    format: str  # csv, json, parquet, api
    size: str
    price: float
    pricing: PricingModel
    last_updated: datetime
    downloads: int = Field(ge=0)
    rating: float = Field(ge=0, le=5)
    quality_score: float = Field(ge=0, le=1)
    visible: bool = True
    schema: Optional[Dict[str, Any]] = None
    preview_available: bool = True

class AgentListing(BaseModel):
    id: str
    name: str
    owner: str
    description: str
    category: str
    capabilities: List[str]
    endpoint: str
    pricing: float
    minimum_stake: float
    rating: float = Field(ge=0, le=5)
    status: str
    last_activity: datetime
    total_earnings: float = Field(ge=0)
    active_services: int = Field(ge=0)
    success_rate: float = Field(ge=0, le=1)

class ServiceRequest(BaseModel):
    id: str
    service_id: str
    requester: str
    provider: str
    agreed_price: float
    escrow_amount: float
    deadline: datetime
    status: ServiceStatus
    parameters: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.now)

class MarketplaceStats(BaseModel):
    total_services: int
    total_data_products: int
    total_agents: int
    active_transactions: int
    total_revenue: float
    average_rating: float
    user_count: int

class UserPreferences(BaseModel):
    categories: List[str] = []
    price_range: Dict[str, float] = Field(default_factory=lambda: {"min": 0, "max": 1000})
    preferred_providers: List[str] = []
    data_formats: List[str] = []
    rating_threshold: float = Field(default=3.0, ge=0, le=5)

class RecommendationRequest(BaseModel):
    preferences: UserPreferences
    context: Optional[Dict[str, Any]] = None
    limit: int = Field(default=10, ge=1, le=50)

class RecommendationResult(BaseModel):
    item_id: str
    item_type: str  # "service" or "data_product"
    match_score: float = Field(ge=0, le=1)
    reason: str

class AnalyticsRequest(BaseModel):
    metric_type: str
    timeframe: str = "7d"
    filters: Optional[Dict[str, Any]] = None

class MarketplaceAnalytics(BaseModel):
    timeframe: str
    total_revenue: float
    transaction_count: int
    user_engagement: Dict[str, Any]
    top_performers: Dict[str, List[Dict[str, Any]]]
    growth_metrics: Dict[str, float]

class WebSocketMessage(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    user_id: Optional[str] = None

class CheckoutItem(BaseModel):
    id: str
    name: str
    type: str  # "service" or "data_product"
    price: float
    provider: str

class Order(BaseModel):
    id: str
    user_id: str
    items: List[CheckoutItem]
    total_amount: float
    status: str
    payment_method: str
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

class QualityMetric(BaseModel):
    metric_type: str
    value: float
    threshold: float
    passed: bool
    calculated_at: datetime

class DataProductMetadata(BaseModel):
    schema_version: str
    source_system: str
    update_frequency: str
    quality_metrics: List[QualityMetric]
    tags: List[str] = []
    compliance_info: Optional[Dict[str, Any]] = None

class ServiceCapability(BaseModel):
    name: str
    description: str
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    examples: List[Dict[str, Any]] = []

class AgentService(BaseModel):
    id: str
    agent_id: str
    name: str
    description: str
    capabilities: List[ServiceCapability]
    pricing: float
    estimated_time: int  # in minutes
    requirements: List[str] = []
    success_rate: float = Field(ge=0, le=1)

class MarketplaceFilter(BaseModel):
    category: Optional[str] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    rating_min: Optional[float] = None
    provider: Optional[str] = None
    search_query: Optional[str] = None
    data_format: Optional[str] = None
    update_frequency: Optional[str] = None
    available_only: bool = True

class PaginatedResponse(BaseModel):
    items: List[Union[Service, DataProduct, AgentListing]]
    total: int
    offset: int
    limit: int
    has_next: bool
    has_previous: bool

class RecommendationEngine(BaseModel):
    user_id: str
    interaction_history: List[Dict[str, Any]] = []
    preferences: UserPreferences
    context: Dict[str, Any] = Field(default_factory=dict)
    
    def calculate_service_score(self, service: Service) -> float:
        """Calculate recommendation score for a service"""
        score = 0.0
        
        # Category preference
        if service.category in self.preferences.categories:
            score += 0.3
        
        # Price preference
        price_range = self.preferences.price_range
        if price_range["min"] <= service.price <= price_range["max"]:
            score += 0.2
        
        # Rating threshold
        if service.rating >= self.preferences.rating_threshold:
            score += 0.2
        
        # Provider preference
        if service.provider in self.preferences.preferred_providers:
            score += 0.15
        
        # Popularity (review count)
        if service.review_count > 50:
            score += 0.1
        
        # Interaction history bonus
        for interaction in self.interaction_history[-10:]:  # Recent interactions
            if interaction.get("category") == service.category:
                score += 0.05
        
        return min(score, 1.0)
    
    def calculate_data_score(self, data_product: DataProduct) -> float:
        """Calculate recommendation score for a data product"""
        score = 0.0
        
        # Category preference
        if data_product.category in self.preferences.categories:
            score += 0.3
        
        # Format preference
        if data_product.format in self.preferences.data_formats:
            score += 0.2
        
        # Quality score
        score += data_product.quality_score * 0.2
        
        # Price preference
        price_range = self.preferences.price_range
        if price_range["min"] <= data_product.price <= price_range["max"]:
            score += 0.15
        
        # Recency bonus
        days_since_update = (datetime.now() - data_product.last_updated).days
        if days_since_update <= 7:
            score += 0.1
        elif days_since_update <= 30:
            score += 0.05
        
        # Download popularity
        if data_product.downloads > 1000:
            score += 0.05
        
        return min(score, 1.0)

class MarketplaceEvent(BaseModel):
    event_type: str
    entity_id: str
    entity_type: str  # "service", "data_product", "agent"
    user_id: str
    event_data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)

class NotificationPreferences(BaseModel):
    new_services: bool = True
    price_changes: bool = True
    service_updates: bool = True
    recommendations: bool = True
    system_alerts: bool = True
    email_enabled: bool = True
    websocket_enabled: bool = True