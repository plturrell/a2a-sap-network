"""
import time
SAP HANA Advanced Features Integration
Implements spatial data processing, graph processing, and advanced analytics
"""

from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum
from app.core.loggingConfig import get_logger, LogCategory
from app.core.exceptions import SecurityError

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from geoalchemy2.functions import ST_Distance, ST_Within

from app.core.config import get_settings

settings = get_settings()
logger = get_logger(__name__, LogCategory.AGENT)


class SpatialDataType(Enum):
    """Spatial data types for financial analytics"""
    POINT = "POINT"
    POLYGON = "POLYGON"
    MULTIPOLYGON = "MULTIPOLYGON"
    LINESTRING = "LINESTRING"


class GraphAlgorithm(Enum):
    """Graph algorithms available in HANA"""
    SHORTEST_PATH = "SHORTEST_PATH"
    CONNECTED_COMPONENTS = "CONNECTED_COMPONENTS"
    PAGERANK = "PAGERANK"
    BETWEENNESS_CENTRALITY = "BETWEENNESS_CENTRALITY"
    CLUSTERING_COEFFICIENT = "CLUSTERING_COEFFICIENT"
    COMMUNITY_DETECTION = "COMMUNITY_DETECTION"


@dataclass
class SpatialQuery:
    """Configuration for spatial queries"""
    geometry_column: str
    spatial_operation: str
    reference_geometry: str
    distance_threshold: float = None
    srid: int = 4326  # WGS84


@dataclass
class GraphAnalysisConfig:
    """Configuration for graph analysis"""
    algorithm: GraphAlgorithm
    vertex_table: str
    edge_table: str
    vertex_key: str
    edge_source: str
    edge_target: str
    weight_column: str = None
    parameters: Dict[str, Any] = None


class HANAAdvancedFeatures:
    """
    SAP HANA Advanced Features Integration
    Provides spatial processing, graph analytics, and advanced functions
    """

    def __init__(self):
        # Validate database URL is properly configured
        db_url = getattr(settings, 'HANA_DATABASE_URL', None)
        if not db_url:
            raise ValueError("HANA_DATABASE_URL must be configured in environment variables")

        # Security: Ensure connection uses encryption
        if not any(param in db_url.lower() for param in ['encrypt=true', 'ssl=true', 'sslmode=require']):
            raise SecurityError("🔒 HANA connections must use encryption for SAP enterprise standards")

        # Create engine with secure connection parameters
        self.engine = create_async_engine(
            db_url,
            echo=False,  # Never log SQL in production (could expose sensitive data)
            future=True,
            pool_size=10,  # Reduced from 20 for better resource management
            max_overflow=20,  # Reduced from 30
            pool_pre_ping=True,  # Test connections before use
            pool_recycle=3600,  # Recycle connections every hour
            connect_args={
                "sslmode": "require",  # Enforce SSL/TLS
                "connect_timeout": 30,
                "application_name": "a2a-hana-advanced"
            }
        )
        self.session_maker = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    async def create_spatial_tables(self):
        """Create spatial tables for financial data geolocation"""

        spatial_ddl = """
        -- Financial institutions location table
        CREATE TABLE FINANCIAL_INSTITUTIONS_SPATIAL (
            INSTITUTION_ID VARCHAR(50) PRIMARY KEY,
            NAME VARCHAR(255),
            TYPE VARCHAR(100), -- BANK, EXCHANGE, REGULATOR, etc.
            COUNTRY VARCHAR(3),
            CITY VARCHAR(100),
            LOCATION ST_GEOMETRY(4326), -- WGS84 coordinate system
            COVERAGE_AREA ST_GEOMETRY(4326),
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Market regions with geographic boundaries
        CREATE TABLE MARKET_REGIONS_SPATIAL (
            REGION_ID VARCHAR(50) PRIMARY KEY,
            NAME VARCHAR(255),
            REGION_TYPE VARCHAR(50), -- COUNTRY, STATE, ECONOMIC_ZONE, etc.
            BOUNDARY ST_GEOMETRY(4326),
            REGULATORY_FRAMEWORK VARCHAR(100),
            CURRENCY VARCHAR(3),
            TIMEZONE VARCHAR(50),
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Trading venues with location data
        CREATE TABLE TRADING_VENUES_SPATIAL (
            VENUE_ID VARCHAR(50) PRIMARY KEY,
            NAME VARCHAR(255),
            VENUE_TYPE VARCHAR(50), -- EXCHANGE, MTF, OTF, etc.
            LOCATION ST_GEOMETRY(4326),
            SERVICE_RADIUS ST_GEOMETRY(4326),
            OPERATING_HOURS VARCHAR(100),
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Risk concentration by geography
        CREATE TABLE RISK_CONCENTRATION_SPATIAL (
            CONCENTRATION_ID VARCHAR(50) PRIMARY KEY,
            PORTFOLIO_ID VARCHAR(50),
            RISK_TYPE VARCHAR(50),
            GEOGRAPHIC_AREA ST_GEOMETRY(4326),
            EXPOSURE_AMOUNT DECIMAL(15,2),
            CONCENTRATION_RATIO DECIMAL(5,4),
            MEASUREMENT_DATE DATE,
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Create spatial indexes
        CREATE INDEX IDX_FI_LOCATION ON FINANCIAL_INSTITUTIONS_SPATIAL(LOCATION);
        CREATE INDEX IDX_MR_BOUNDARY ON MARKET_REGIONS_SPATIAL(BOUNDARY);
        CREATE INDEX IDX_TV_LOCATION ON TRADING_VENUES_SPATIAL(LOCATION);
        CREATE INDEX IDX_RC_AREA ON RISK_CONCENTRATION_SPATIAL(GEOGRAPHIC_AREA);
        """

        async with self.session_maker() as session:
            try:
                for statement in spatial_ddl.split(';'):
                    if statement.strip():
                        await session.execute(text(statement))
                await session.commit()
                logger.info("Spatial tables created successfully")
            except Exception as e:
                await session.rollback()
                if "already exists" not in str(e).lower():
                    logger.error(f"Error creating spatial tables: {e}")
                    raise

    async def create_graph_tables(self):
        """Create graph tables for relationship analysis"""

        graph_ddl = """
        -- Financial entity relationships graph
        CREATE TABLE FINANCIAL_ENTITIES_VERTICES (
            ENTITY_ID VARCHAR(50) PRIMARY KEY,
            ENTITY_TYPE VARCHAR(50), -- BANK, COMPANY, FUND, PERSON, etc.
            NAME VARCHAR(255),
            COUNTRY VARCHAR(3),
            SECTOR VARCHAR(100),
            MARKET_CAP DECIMAL(15,2),
            RISK_RATING VARCHAR(10),
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE FINANCIAL_RELATIONSHIPS_EDGES (
            EDGE_ID VARCHAR(50) PRIMARY KEY,
            SOURCE_ENTITY VARCHAR(50),
            TARGET_ENTITY VARCHAR(50),
            RELATIONSHIP_TYPE VARCHAR(50), -- OWNS, TRADES_WITH, LENDS_TO, etc.
            RELATIONSHIP_STRENGTH DECIMAL(3,2),
            TRANSACTION_VOLUME DECIMAL(15,2),
            RISK_IMPACT DECIMAL(3,2),
            START_DATE DATE,
            END_DATE DATE,
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (SOURCE_ENTITY) REFERENCES FINANCIAL_ENTITIES_VERTICES(ENTITY_ID),
            FOREIGN KEY (TARGET_ENTITY) REFERENCES FINANCIAL_ENTITIES_VERTICES(ENTITY_ID)
        );

        -- Market correlation network
        CREATE TABLE MARKET_INSTRUMENTS_VERTICES (
            INSTRUMENT_ID VARCHAR(50) PRIMARY KEY,
            SYMBOL VARCHAR(20),
            NAME VARCHAR(255),
            ASSET_CLASS VARCHAR(50),
            MARKET VARCHAR(50),
            SECTOR VARCHAR(100),
            MARKET_CAP DECIMAL(15,2),
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE MARKET_CORRELATIONS_EDGES (
            CORRELATION_ID VARCHAR(50) PRIMARY KEY,
            SOURCE_INSTRUMENT VARCHAR(50),
            TARGET_INSTRUMENT VARCHAR(50),
            CORRELATION_COEFFICIENT DECIMAL(8,6),
            CORRELATION_TYPE VARCHAR(50), -- PRICE, VOLATILITY, VOLUME, etc.
            TIME_WINDOW_DAYS INTEGER,
            CALCULATION_DATE DATE,
            SIGNIFICANCE_LEVEL DECIMAL(4,3),
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (SOURCE_INSTRUMENT) REFERENCES MARKET_INSTRUMENTS_VERTICES(INSTRUMENT_ID),
            FOREIGN KEY (TARGET_INSTRUMENT) REFERENCES MARKET_INSTRUMENTS_VERTICES(INSTRUMENT_ID)
        );

        -- Create graph workspace
        CREATE GRAPH WORKSPACE FINANCIAL_NETWORK
            EDGE TABLE FINANCIAL_RELATIONSHIPS_EDGES
            SOURCE COLUMN SOURCE_ENTITY
            TARGET COLUMN TARGET_ENTITY
            KEY COLUMN EDGE_ID
            VERTEX TABLE FINANCIAL_ENTITIES_VERTICES
            KEY COLUMN ENTITY_ID;

        CREATE GRAPH WORKSPACE MARKET_CORRELATION_NETWORK
            EDGE TABLE MARKET_CORRELATIONS_EDGES
            SOURCE COLUMN SOURCE_INSTRUMENT
            TARGET COLUMN TARGET_INSTRUMENT
            KEY COLUMN CORRELATION_ID
            VERTEX TABLE MARKET_INSTRUMENTS_VERTICES
            KEY COLUMN INSTRUMENT_ID;
        """

        async with self.session_maker() as session:
            try:
                for statement in graph_ddl.split(';'):
                    if statement.strip():
                        await session.execute(text(statement))
                await session.commit()
                logger.info("Graph tables and workspaces created successfully")
            except Exception as e:
                await session.rollback()
                if "already exists" not in str(e).lower():
                    logger.error(f"Error creating graph tables: {e}")
                    raise

    async def spatial_risk_analysis(self, portfolio_id: str,
                                  risk_threshold: float = 0.1) -> Dict[str, Any]:
        """Analyze geographic risk concentration using spatial functions"""

        spatial_query = """
        SELECT
            r.REGION_ID,
            r.NAME as REGION_NAME,
            r.CURRENCY,
            SUM(rc.EXPOSURE_AMOUNT) as TOTAL_EXPOSURE,
            AVG(rc.CONCENTRATION_RATIO) as AVG_CONCENTRATION,
            COUNT(*) as NUM_POSITIONS,
            ST_Area(r.BOUNDARY) as REGION_AREA_SQM
        FROM MARKET_REGIONS_SPATIAL r
        JOIN RISK_CONCENTRATION_SPATIAL rc
            ON ST_Within(rc.GEOGRAPHIC_AREA, r.BOUNDARY) = 1
        WHERE rc.PORTFOLIO_ID = :portfolio_id
            AND rc.CONCENTRATION_RATIO > :risk_threshold
        GROUP BY r.REGION_ID, r.NAME, r.CURRENCY, r.BOUNDARY
        ORDER BY TOTAL_EXPOSURE DESC
        """

        async with self.session_maker() as session:
            result = await session.execute(
                text(spatial_query),
                {
                    "portfolio_id": portfolio_id,
                    "risk_threshold": risk_threshold
                }
            )

            risk_concentrations = []
            for row in result:
                risk_concentrations.append({
                    "region_id": row.REGION_ID,
                    "region_name": row.REGION_NAME,
                    "currency": row.CURRENCY,
                    "total_exposure": float(row.TOTAL_EXPOSURE),
                    "avg_concentration": float(row.AVG_CONCENTRATION),
                    "num_positions": row.NUM_POSITIONS,
                    "region_area_sqm": float(row.REGION_AREA_SQM)
                })

        return {
            "portfolio_id": portfolio_id,
            "risk_threshold": risk_threshold,
            "analysis_date": datetime.utcnow().isoformat(),
            "geographic_concentrations": risk_concentrations,
            "total_regions": len(risk_concentrations),
            "max_exposure": max([r["total_exposure"] for r in risk_concentrations] or [0])
        }

    async def find_nearby_financial_institutions(self,
                                               latitude: float,
                                               longitude: float,
                                               radius_km: float = 50) -> List[Dict[str, Any]]:
        """Find financial institutions within specified radius using spatial search"""

        spatial_search = """
        SELECT
            INSTITUTION_ID,
            NAME,
            TYPE,
            COUNTRY,
            CITY,
            ST_Distance(LOCATION, ST_GeomFromText('POINT(:longitude :latitude)', 4326), 'kilometer') as DISTANCE_KM
        FROM FINANCIAL_INSTITUTIONS_SPATIAL
        WHERE ST_DWithin(LOCATION, ST_GeomFromText('POINT(:longitude :latitude)', 4326), :radius_km, 'kilometer') = 1
        ORDER BY DISTANCE_KM ASC
        """

        async with self.session_maker() as session:
            result = await session.execute(
                text(spatial_search),
                {
                    "latitude": latitude,
                    "longitude": longitude,
                    "radius_km": radius_km
                }
            )

            institutions = []
            for row in result:
                institutions.append({
                    "institution_id": row.INSTITUTION_ID,
                    "name": row.NAME,
                    "type": row.TYPE,
                    "country": row.COUNTRY,
                    "city": row.CITY,
                    "distance_km": float(row.DISTANCE_KM)
                })

        return institutions

    async def analyze_market_network_centrality(self) -> Dict[str, Any]:
        """Analyze market network using graph algorithms"""

        # PageRank analysis for market influence
        pagerank_query = """
        SELECT
            v.INSTRUMENT_ID,
            v.SYMBOL,
            v.NAME,
            v.ASSET_CLASS,
            v.MARKET,
            pr.PAGERANK_VALUE
        FROM
            GRAPH_TABLE (
                MARKET_CORRELATION_NETWORK,
                'PAGERANK',
                'RESULT_VIEW'
            ) pr
        JOIN MARKET_INSTRUMENTS_VERTICES v ON pr.VERTEX = v.INSTRUMENT_ID
        ORDER BY pr.PAGERANK_VALUE DESC
        LIMIT 50
        """

        # Betweenness centrality for systemic risk identification
        betweenness_query = """
        SELECT
            v.INSTRUMENT_ID,
            v.SYMBOL,
            v.NAME,
            bc.BETWEENNESS_CENTRALITY
        FROM
            GRAPH_TABLE (
                MARKET_CORRELATION_NETWORK,
                'BETWEENNESS_CENTRALITY',
                'RESULT_VIEW'
            ) bc
        JOIN MARKET_INSTRUMENTS_VERTICES v ON bc.VERTEX = v.INSTRUMENT_ID
        ORDER BY bc.BETWEENNESS_CENTRALITY DESC
        LIMIT 20
        """

        async with self.session_maker() as session:
            # Execute PageRank analysis
            pagerank_result = await session.execute(text(pagerank_query))
            market_influence = []
            for row in pagerank_result:
                market_influence.append({
                    "instrument_id": row.INSTRUMENT_ID,
                    "symbol": row.SYMBOL,
                    "name": row.NAME,
                    "asset_class": row.ASSET_CLASS,
                    "market": row.MARKET,
                    "pagerank_value": float(row.PAGERANK_VALUE)
                })

            # Execute betweenness centrality analysis
            betweenness_result = await session.execute(text(betweenness_query))
            systemic_risk_indicators = []
            for row in betweenness_result:
                systemic_risk_indicators.append({
                    "instrument_id": row.INSTRUMENT_ID,
                    "symbol": row.SYMBOL,
                    "name": row.NAME,
                    "betweenness_centrality": float(row.BETWEENNESS_CENTRALITY)
                })

        return {
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "market_influence_ranking": market_influence,
            "systemic_risk_indicators": systemic_risk_indicators,
            "network_metrics": {
                "total_analyzed_instruments": len(market_influence),
                "high_centrality_instruments": len([x for x in systemic_risk_indicators if x["betweenness_centrality"] > 0.1])
            }
        }

    async def detect_financial_network_communities(self) -> Dict[str, Any]:
        """Detect communities in financial entity network"""

        community_query = """
        SELECT
            v.ENTITY_ID,
            v.NAME,
            v.ENTITY_TYPE,
            v.SECTOR,
            v.COUNTRY,
            cd.COMMUNITY_ID,
            COUNT(*) OVER (PARTITION BY cd.COMMUNITY_ID) as COMMUNITY_SIZE
        FROM
            GRAPH_TABLE (
                FINANCIAL_NETWORK,
                'LOUVAIN_COMMUNITY_DETECTION',
                'RESULT_VIEW'
            ) cd
        JOIN FINANCIAL_ENTITIES_VERTICES v ON cd.VERTEX = v.ENTITY_ID
        ORDER BY cd.COMMUNITY_ID, v.NAME
        """

        async with self.session_maker() as session:
            result = await session.execute(text(community_query))

            communities = {}
            for row in result:
                community_id = row.COMMUNITY_ID
                if community_id not in communities:
                    communities[community_id] = {
                        "community_id": community_id,
                        "size": row.COMMUNITY_SIZE,
                        "members": []
                    }

                communities[community_id]["members"].append({
                    "entity_id": row.ENTITY_ID,
                    "name": row.NAME,
                    "entity_type": row.ENTITY_TYPE,
                    "sector": row.SECTOR,
                    "country": row.COUNTRY
                })

        # Calculate community statistics
        community_stats = []
        for community in communities.values():
            sectors = [m["sector"] for m in community["members"] if m["sector"]]
            countries = [m["country"] for m in community["members"] if m["country"]]

            community_stats.append({
                "community_id": community["community_id"],
                "size": community["size"],
                "dominant_sector": max(set(sectors), key=sectors.count) if sectors else None,
                "geographic_diversity": len(set(countries)),
                "members": community["members"]
            })

        return {
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "total_communities": len(communities),
            "communities": community_stats,
            "network_modularity": self._calculate_modularity(community_stats)
        }

    async def shortest_path_analysis(self, source_entity: str,
                                   target_entity: str) -> Dict[str, Any]:
        """Find shortest path between financial entities"""

        path_query = """
        SELECT
            sp.EDGE_ORDER,
            e.EDGE_ID,
            e.SOURCE_ENTITY,
            e.TARGET_ENTITY,
            e.RELATIONSHIP_TYPE,
            e.RELATIONSHIP_STRENGTH,
            e.TRANSACTION_VOLUME,
            vs.NAME as SOURCE_NAME,
            vt.NAME as TARGET_NAME
        FROM
            GRAPH_TABLE (
                FINANCIAL_NETWORK,
                'SHORTEST_PATH',
                'RESULT_VIEW',
                'START_VERTEX', :source_entity,
                'END_VERTEX', :target_entity
            ) sp
        JOIN FINANCIAL_RELATIONSHIPS_EDGES e ON sp.EDGE_ID = e.EDGE_ID
        JOIN FINANCIAL_ENTITIES_VERTICES vs ON e.SOURCE_ENTITY = vs.ENTITY_ID
        JOIN FINANCIAL_ENTITIES_VERTICES vt ON e.TARGET_ENTITY = vt.ENTITY_ID
        ORDER BY sp.EDGE_ORDER
        """

        async with self.session_maker() as session:
            result = await session.execute(
                text(path_query),
                {
                    "source_entity": source_entity,
                    "target_entity": target_entity
                }
            )

            path_edges = []
            total_strength = 0
            total_volume = 0

            for row in result:
                edge_info = {
                    "order": row.EDGE_ORDER,
                    "edge_id": row.EDGE_ID,
                    "source_entity": row.SOURCE_ENTITY,
                    "target_entity": row.TARGET_ENTITY,
                    "source_name": row.SOURCE_NAME,
                    "target_name": row.TARGET_NAME,
                    "relationship_type": row.RELATIONSHIP_TYPE,
                    "relationship_strength": float(row.RELATIONSHIP_STRENGTH),
                    "transaction_volume": float(row.TRANSACTION_VOLUME or 0)
                }
                path_edges.append(edge_info)
                total_strength += edge_info["relationship_strength"]
                total_volume += edge_info["transaction_volume"]

        return {
            "source_entity": source_entity,
            "target_entity": target_entity,
            "path_found": len(path_edges) > 0,
            "path_length": len(path_edges),
            "path_edges": path_edges,
            "total_relationship_strength": total_strength,
            "total_transaction_volume": total_volume,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }

    async def time_series_forecast_with_hana(self,
                                           instrument_id: str,
                                           forecast_horizon: int = 30) -> Dict[str, Any]:
        """Time series forecasting using HANA PAL algorithms"""

        # Auto ARIMA forecasting
        forecast_query = """
        CREATE LOCAL TEMPORARY COLUMN TABLE #FORECAST_INPUT AS (
            SELECT
                ROW_NUMBER() OVER (ORDER BY PRICE_DATE) as ID,
                PRICE_DATE,
                CLOSE_PRICE
            FROM PRICE_DATA
            WHERE INSTRUMENT_ID = :instrument_id
                AND PRICE_DATE >= ADD_DAYS(CURRENT_DATE, -365)
            ORDER BY PRICE_DATE
        );

        CALL _SYS_AFL.PAL_AUTO_ARIMA(
            #FORECAST_INPUT,
            (SELECT * FROM #FORECAST_INPUT),
            FORECAST_RESULT,
            'FORECAST_LENGTH', :forecast_horizon
        ) WITH OVERVIEW;

        SELECT * FROM FORECAST_RESULT ORDER BY ID;
        """

        async with self.session_maker() as session:
            try:
                result = await session.execute(
                    text(forecast_query),
                    {
                        "instrument_id": instrument_id,
                        "forecast_horizon": forecast_horizon
                    }
                )

                forecasts = []
                for row in result:
                    forecasts.append({
                        "forecast_id": row.ID,
                        "forecast_date": row.PRICE_DATE.isoformat() if row.PRICE_DATE else None,
                        "forecasted_value": float(row.CLOSE_PRICE),
                        "confidence_interval_lower": float(getattr(row, 'CI_LOWER', 0)),
                        "confidence_interval_upper": float(getattr(row, 'CI_UPPER', 0))
                    })

                return {
                    "instrument_id": instrument_id,
                    "forecast_horizon_days": forecast_horizon,
                    "forecasts": forecasts,
                    "model_type": "AUTO_ARIMA",
                    "generated_at": datetime.utcnow().isoformat()
                }

            except Exception as e:
                logger.error(f"Time series forecasting failed: {e}")
                return {
                    "instrument_id": instrument_id,
                    "error": str(e),
                    "forecasts": []
                }

    def _calculate_modularity(self, communities: List[Dict[str, Any]]) -> float:
        """Calculate network modularity score"""
        total_entities = sum(c["size"] for c in communities)
        if total_entities == 0:
            return 0.0

        # Simplified modularity calculation
        # In practice, this would use the actual edge weights and degrees
        modularity = sum((c["size"] / total_entities) ** 2 for c in communities)
        return round(1 - modularity, 4)

    async def create_sample_data(self):
        """Create sample spatial and graph data for testing"""

        sample_data_sql = """
        -- Sample financial institutions
        INSERT INTO FINANCIAL_INSTITUTIONS_SPATIAL VALUES
        ('FI001', 'Deutsche Bank', 'BANK', 'DEU', 'Frankfurt', ST_GeomFromText('POINT(8.6821 50.1109)', 4326), NULL, CURRENT_TIMESTAMP),
        ('FI002', 'JPMorgan Chase', 'BANK', 'USA', 'New York', ST_GeomFromText('POINT(-74.0059 40.7128)', 4326), NULL, CURRENT_TIMESTAMP),
        ('FI003', 'HSBC', 'BANK', 'GBR', 'London', ST_GeomFromText('POINT(-0.1276 51.5074)', 4326), NULL, CURRENT_TIMESTAMP);

        -- Sample market regions
        INSERT INTO MARKET_REGIONS_SPATIAL VALUES
        ('MR001', 'European Union', 'ECONOMIC_ZONE', ST_GeomFromText('POLYGON(((-10 35), (30 35), (30 70), (-10 70), (-10 35)))', 4326), 'MiFID II', 'EUR', 'CET', CURRENT_TIMESTAMP),
        ('MR002', 'North America', 'REGION', ST_GeomFromText('POLYGON(((-130 25), (-60 25), (-60 55), (-130 55), (-130 25)))', 4326), 'Dodd-Frank', 'USD', 'EST', CURRENT_TIMESTAMP);

        -- Sample financial entities for graph
        INSERT INTO FINANCIAL_ENTITIES_VERTICES VALUES
        ('ENTITY001', 'BANK', 'Deutsche Bank AG', 'DEU', 'Banking', 25000000000, 'A-', CURRENT_TIMESTAMP),
        ('ENTITY002', 'BANK', 'JPMorgan Chase & Co', 'USA', 'Banking', 400000000000, 'AA-', CURRENT_TIMESTAMP),
        ('ENTITY003', 'COMPANY', 'BMW AG', 'DEU', 'Automotive', 50000000000, 'A', CURRENT_TIMESTAMP);

        -- Sample relationships
        INSERT INTO FINANCIAL_RELATIONSHIPS_EDGES VALUES
        ('EDGE001', 'ENTITY001', 'ENTITY003', 'LENDS_TO', 0.8, 1000000000, 0.3, '2024-01-01', NULL, CURRENT_TIMESTAMP),
        ('EDGE002', 'ENTITY002', 'ENTITY001', 'TRADES_WITH', 0.9, 5000000000, 0.2, '2024-01-01', NULL, CURRENT_TIMESTAMP);
        """

        async with self.session_maker() as session:
            try:
                for statement in sample_data_sql.split(';'):
                    if statement.strip():
                        await session.execute(text(statement))
                await session.commit()
                logger.info("Sample spatial and graph data created successfully")
            except Exception as e:
                await session.rollback()
                logger.error(f"Error creating sample data: {e}")


# Integration with A2A agents
class HANAAdvancedIntegration:
    """Integration of HANA advanced features with A2A agents"""

    def __init__(self):
        self.hana_features = HANAAdvancedFeatures()

    async def enhance_agent3_vector_processing(self, vector_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance Agent 3 vector processing with spatial and graph context"""

        enhanced_data = vector_data.copy()

        # Add spatial context if location data is available
        if "location" in vector_data:
            lat, lon = vector_data["location"]["latitude"], vector_data["location"]["longitude"]
            nearby_institutions = await self.hana_features.find_nearby_financial_institutions(lat, lon)
            enhanced_data["spatial_context"] = {
                "nearby_institutions": nearby_institutions,
                "location_risk_factors": len(nearby_institutions)
            }

        # Add network centrality if entity relationships exist
        if "entity_id" in vector_data:
            try:
                network_analysis = await self.hana_features.analyze_market_network_centrality()
                enhanced_data["network_context"] = network_analysis
            except Exception as e:
                logger.warning(f"Network analysis failed: {e}")

        return enhanced_data

    async def enhance_agent4_calculations(self, calculation_request: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance Agent 4 calculations with advanced HANA analytics"""

        enhanced_request = calculation_request.copy()

        # Add spatial risk analysis for geographic concentration
        if "portfolio_id" in calculation_request:
            spatial_risk = await self.hana_features.spatial_risk_analysis(
                calculation_request["portfolio_id"]
            )
            enhanced_request["spatial_risk_analysis"] = spatial_risk

        # Add time series forecasting for predictive analytics
        if "instruments" in calculation_request:
            forecasts = {}
            for instrument_id in calculation_request["instruments"][:5]:  # Limit to 5 for performance
                try:
                    forecast = await self.hana_features.time_series_forecast_with_hana(instrument_id)
                    forecasts[instrument_id] = forecast
                except Exception as e:
                    logger.warning(f"Forecast failed for {instrument_id}: {e}")

            enhanced_request["predictive_analytics"] = forecasts

        return enhanced_request