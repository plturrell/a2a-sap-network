from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class ResourceType(str, Enum):
    DATA_PRODUCT = "dataProduct"
    API = "api"
    EVENT = "event"
    ENTITY_TYPE = "entityType"


class RegistrationStatus(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


class DublinCoreMetadata(BaseModel):
    """Dublin Core metadata elements - ISO 15836, RFC 5013 compliant"""
    title: Optional[str] = Field(None, description="A name given to the resource")
    creator: Optional[List[str]] = Field(None, description="Entity primarily responsible for making the resource")
    subject: Optional[List[str]] = Field(None, description="Topic of the resource")
    description: Optional[str] = Field(None, description="An account of the resource")
    publisher: Optional[str] = Field(None, description="Entity responsible for making the resource available")
    contributor: Optional[List[str]] = Field(None, description="Entity responsible for making contributions")
    date: Optional[str] = Field(None, description="Point or period of time associated with an event in the lifecycle")
    type: Optional[str] = Field(None, description="Nature or genre of the resource")
    format: Optional[str] = Field(None, description="File format, physical medium, or dimensions")
    identifier: Optional[str] = Field(None, description="Unambiguous reference to the resource")
    source: Optional[str] = Field(None, description="Related resource from which the described resource is derived")
    language: Optional[str] = Field(None, description="Language of the resource")
    relation: Optional[List[str]] = Field(None, description="Related resources")
    coverage: Optional[str] = Field(None, description="Spatial or temporal topic of the resource")
    rights: Optional[str] = Field(None, description="Information about rights held in and over the resource")


class ORDDocument(BaseModel):
    """ORD Document structure according to SAP ORD specification with Dublin Core enhancement"""
    openResourceDiscovery: str = Field(default="1.5.0")
    description: Optional[str] = None
    dublinCore: Optional[DublinCoreMetadata] = None
    resources: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    apiResources: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    entityTypes: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    eventResources: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    dataProducts: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    cdsSchema: Optional[Dict[str, Any]] = None  # For CDS CSN storage


class DublinCoreQualityMetrics(BaseModel):
    """Dublin Core quality assessment metrics"""
    completeness: float = Field(0.0, ge=0.0, le=1.0, description="Percentage of populated elements")
    accuracy: float = Field(0.0, ge=0.0, le=1.0, description="Format compliance and semantic correctness")
    consistency: float = Field(0.0, ge=0.0, le=1.0, description="Cross-element coherence")
    timeliness: float = Field(0.0, ge=0.0, le=1.0, description="Currency of metadata")
    overall_score: float = Field(0.0, ge=0.0, le=1.0, description="Overall quality score")
    iso15836_compliant: bool = Field(False, description="ISO 15836 compliance")
    rfc5013_compliant: bool = Field(False, description="RFC 5013 compliance")
    ansi_niso_compliant: bool = Field(False, description="ANSI/NISO Z39.85 compliance")


class ValidationResult(BaseModel):
    valid: bool
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    compliance_score: float = 0.0
    dublincore_validation: Optional[DublinCoreQualityMetrics] = None


class RegistrationMetadata(BaseModel):
    registered_by: str
    registered_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    status: RegistrationStatus = RegistrationStatus.ACTIVE


class GovernanceInfo(BaseModel):
    owner: Optional[str] = None
    steward: Optional[str] = None
    classification: Optional[str] = None
    retention_policy: Optional[str] = None


class AnalyticsInfo(BaseModel):
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    popular_resources: List[str] = Field(default_factory=list)
    dublin_core_search_hits: int = 0
    facet_usage_stats: Dict[str, int] = Field(default_factory=dict)


class ORDRegistration(BaseModel):
    """Complete ORD Registration record"""
    registration_id: str
    ord_document: ORDDocument
    metadata: RegistrationMetadata
    validation: ValidationResult
    governance: GovernanceInfo = Field(default_factory=GovernanceInfo)
    analytics: AnalyticsInfo = Field(default_factory=AnalyticsInfo)


class ResourceIndexEntry(BaseModel):
    """Searchable index entry for ORD resources with Dublin Core"""
    ord_id: str
    registration_id: str
    resource_type: ResourceType
    title: str
    short_description: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    labels: Dict[str, str] = Field(default_factory=dict)
    domain: Optional[str] = None
    category: Optional[str] = None
    access_strategies: List[Dict[str, Any]] = Field(default_factory=list)
    compliance_info: Dict[str, Any] = Field(default_factory=dict)
    indexed_at: datetime = Field(default_factory=datetime.utcnow)
    searchable_content: Optional[str] = None
    # Dublin Core fields for enhanced search
    dublin_core: Optional[DublinCoreMetadata] = None
    dc_creator: Optional[List[str]] = None
    dc_subject: Optional[List[str]] = None
    dc_publisher: Optional[str] = None
    dc_format: Optional[str] = None


class RegistrationRequest(BaseModel):
    """Request to register an ORD document"""
    ord_document: ORDDocument
    registered_by: str
    tags: Optional[List[str]] = None
    labels: Optional[Dict[str, str]] = None


class RegistrationResponse(BaseModel):
    """Response from registration"""
    registration_id: str
    status: str
    validation_results: ValidationResult
    registered_at: datetime
    registry_url: str


class SearchRequest(BaseModel):
    """Search request parameters with Dublin Core support"""
    query: Optional[str] = None
    resource_type: Optional[ResourceType] = None
    resource_types: Optional[List[ResourceType]] = None  # For backward compatibility
    tags: Optional[List[str]] = None
    domain: Optional[str] = None
    category: Optional[str] = None
    # Dublin Core search facets
    creator: Optional[str] = None
    subject: Optional[str] = None
    publisher: Optional[str] = None
    format: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None  # Missing filters field
    includeDublinCore: bool = Field(default=True)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, le=100)


class SearchFacet(BaseModel):
    """Search facet for Dublin Core fields"""
    value: str
    count: int


class SearchResult(BaseModel):
    """Search result with Dublin Core facets"""
    results: List[ResourceIndexEntry]
    total_count: int
    page: int
    page_size: int
    facets: Optional[Dict[str, List[SearchFacet]]] = None


class DublinCoreValidationRequest(BaseModel):
    """Request to validate Dublin Core metadata"""
    dublin_core: DublinCoreMetadata
    strict_mode: bool = Field(default=False, description="Enforce strict validation")


class DublinCoreValidationResponse(BaseModel):
    """Dublin Core validation response"""
    valid: bool
    quality_metrics: DublinCoreQualityMetrics
    metadata_completeness: float
    recommendations: List[str] = Field(default_factory=list)
