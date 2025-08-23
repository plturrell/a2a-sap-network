"""
S/4HANA Integration for A2A Network
Specifically designed for the Agent-to-Agent financial data processing system
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
from dataclasses import dataclass

from app.core.logger import get_logger
from app.core.sapCloudSdk import SAPCloudSDK
from app.core.sapGraphClient import SAPGraphClient
from app.core.metrics import track_performance, MetricsCollector
from app.a2a.models import AgentMessage, WorkflowData, DataProduct
from app.a2a.messageQueue import MessageQueueService

logger = get_logger(__name__)
metrics = MetricsCollector()


class S4HANADataType(Enum):
    """S/4HANA data types relevant to A2A Network"""
    FINANCIAL_STATEMENT = "financial_statement"
    COST_CENTER = "cost_center"
    PROFIT_CENTER = "profit_center"
    GL_ACCOUNT = "gl_account"
    JOURNAL_ENTRY = "journal_entry"
    FINANCIAL_PLAN = "financial_plan"
    BUDGET = "budget"
    ANALYTICS = "analytics"
    CONSOLIDATION = "consolidation"
    REGULATORY_REPORT = "regulatory_report"


@dataclass
class S4HANADataProduct:
    """S/4HANA data product for A2A processing"""
    id: str
    type: S4HANADataType
    company_code: str
    fiscal_year: int
    fiscal_period: Optional[int]
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    source_system: str = "S4HANA"
    timestamp: datetime = None
    
    def to_a2a_data_product(self) -> DataProduct:
        """Convert to A2A DataProduct format"""
        return DataProduct(
            id=self.id,
            title=f"{self.type.value}_{self.company_code}_{self.fiscal_year}",
            description=f"S/4HANA {self.type.value} data",
            format="application/json",
            source=self.source_system,
            created=self.timestamp or datetime.utcnow(),
            metadata={
                **self.metadata,
                "s4hana_type": self.type.value,
                "company_code": self.company_code,
                "fiscal_year": self.fiscal_year,
                "fiscal_period": self.fiscal_period
            },
            data=self.data
        )


class S4HANAA2AIntegration:
    """
    S/4HANA integration specifically for A2A Network
    Handles financial data extraction and agent workflow integration
    """
    
    def __init__(self):
        self.sap_sdk = SAPCloudSDK()
        self.sap_graph = SAPGraphClient()
        self.message_queue = MessageQueueService()
        self._session: Optional[aiohttp.ClientSession] = None
        
        # S/4HANA OData v4 endpoints for financial data
        self.FINANCIAL_SERVICES = {
            "financial_statement": "/sap/opu/odata4/sap/api_finstatement/srvd_a2x/sap/finstatement/0001",
            "cost_center": "/sap/opu/odata4/sap/api_costcenter/srvd_a2x/sap/costcenter/0001",
            "profit_center": "/sap/opu/odata4/sap/api_profitcenter/srvd_a2x/sap/profitcenter/0001",
            "gl_account": "/sap/opu/odata4/sap/api_glaccountlineitem/srvd_a2x/sap/glaccountlineitem/0001",
            "journal_entry": "/sap/opu/odata4/sap/api_journalentry/srvd_a2x/sap/journalentry/0001",
            "financial_plan": "/sap/opu/odata4/sap/api_finplan/srvd_a2x/sap/finplan/0001",
            "budget": "/sap/opu/odata4/sap/api_budget/srvd_a2x/sap/budget/0001",
            "analytics": "/sap/opu/odata4/sap/api_finanalytics/srvd_a2x/sap/finanalytics/0001"
        }
        
        # Agent mappings for S/4HANA data types
        self.AGENT_ROUTING = {
            S4HANADataType.FINANCIAL_STATEMENT: 0,  # Data Product Agent
            S4HANADataType.COST_CENTER: 0,
            S4HANADataType.PROFIT_CENTER: 0,
            S4HANADataType.GL_ACCOUNT: 0,
            S4HANADataType.JOURNAL_ENTRY: 4,  # Calculation Validation Agent
            S4HANADataType.FINANCIAL_PLAN: 9,  # Reasoning Agent
            S4HANADataType.BUDGET: 10,  # Calculation Agent
            S4HANADataType.ANALYTICS: 3,  # Vector Processing Agent
            S4HANADataType.CONSOLIDATION: 15,  # Orchestrator Agent
            S4HANADataType.REGULATORY_REPORT: 5  # QA Validation Agent
        }
    
    async def connect(self):
        """Initialize connection to S/4HANA"""
        try:
            # Use SAP SDK for authentication
            await self.sap_sdk.initialize()
            
            # Create session with SAP headers
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "DataServiceVersion": "4.0",
                "MaxDataServiceVersion": "4.0"
            }
            
            # Get authentication from SDK
            auth_headers = await self.sap_sdk.get_auth_headers()
            headers.update(auth_headers)
            
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout
            )
            
            logger.info("Connected to S/4HANA for A2A Network")
            metrics.increment("s4hana.a2a.connections.success")
            
        except Exception as e:
            logger.error(f"Failed to connect to S/4HANA: {e}")
            metrics.increment("s4hana.a2a.connections.failure")
            raise
    
    async def disconnect(self):
        """Close S/4HANA connection"""
        if self._session:
            await self._session.close()
    
    @track_performance("s4hana_extract_financial_data")
    async def extract_financial_data(
        self,
        data_type: S4HANADataType,
        company_code: str,
        fiscal_year: int,
        fiscal_period: Optional[int] = None,
        additional_filters: Optional[Dict[str, Any]] = None
    ) -> S4HANADataProduct:
        """
        Extract financial data from S/4HANA for A2A processing
        """
        try:
            # Build filter conditions
            filters = [
                f"CompanyCode eq '{company_code}'",
                f"FiscalYear eq {fiscal_year}"
            ]
            
            if fiscal_period:
                filters.append(f"FiscalPeriod eq {fiscal_period:03d}")
            
            if additional_filters:
                for key, value in additional_filters.items():
                    if isinstance(value, str):
                        filters.append(f"{key} eq '{value}'")
                    else:
                        filters.append(f"{key} eq {value}")
            
            filter_str = " and ".join(filters)
            
            # Get data based on type
            if data_type == S4HANADataType.FINANCIAL_STATEMENT:
                data = await self._get_financial_statements(filter_str)
            elif data_type == S4HANADataType.COST_CENTER:
                data = await self._get_cost_center_data(filter_str)
            elif data_type == S4HANADataType.PROFIT_CENTER:
                data = await self._get_profit_center_data(filter_str)
            elif data_type == S4HANADataType.GL_ACCOUNT:
                data = await self._get_gl_account_data(filter_str)
            elif data_type == S4HANADataType.JOURNAL_ENTRY:
                data = await self._get_journal_entries(filter_str)
            elif data_type == S4HANADataType.FINANCIAL_PLAN:
                data = await self._get_financial_plan_data(filter_str)
            elif data_type == S4HANADataType.BUDGET:
                data = await self._get_budget_data(filter_str)
            elif data_type == S4HANADataType.ANALYTICS:
                data = await self._get_analytics_data(company_code, fiscal_year)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            
            # Create S4HANA data product
            product = S4HANADataProduct(
                id=f"s4h_{data_type.value}_{company_code}_{fiscal_year}_{fiscal_period or 'YTD'}",
                type=data_type,
                company_code=company_code,
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period,
                data=data,
                metadata={
                    "extraction_timestamp": datetime.utcnow().isoformat(),
                    "record_count": len(data.get("items", [])),
                    "filters_applied": filter_str
                },
                timestamp=datetime.utcnow()
            )
            
            metrics.increment("s4hana.a2a.extractions.success", tags={"type": data_type.value})
            return product
            
        except Exception as e:
            logger.error(f"Failed to extract {data_type.value} data: {e}")
            metrics.increment("s4hana.a2a.extractions.failure", tags={"type": data_type.value})
            raise
    
    async def _get_financial_statements(self, filter_str: str) -> Dict[str, Any]:
        """Get financial statement data"""
        # Query financial statement items
        endpoint = f"{self.FINANCIAL_SERVICES['financial_statement']}/FinancialStatementItem"
        
        params = {
            "$filter": filter_str,
            "$select": "GLAccount,GLAccountName,AmountInCompanyCodeCurrency,Currency,DebitCreditCode,FinancialStatementItem,HierarchyNode",
            "$orderby": "FinancialStatementItem"
        }
        
        async with self._session.get(endpoint, params=params) as response:
            response.raise_for_status()
            result = await response.json()
            
        return {
            "statement_type": "balance_sheet_pl",
            "items": result.get("value", []),
            "currency": result.get("value", [{}])[0].get("Currency", "EUR") if result.get("value") else "EUR"
        }
    
    async def _get_cost_center_data(self, filter_str: str) -> Dict[str, Any]:
        """Get cost center actuals and plan data"""
        endpoint = f"{self.FINANCIAL_SERVICES['cost_center']}/CostCenterActualPlan"
        
        params = {
            "$filter": filter_str,
            "$select": "CostCenter,CostCenterName,CostElement,ActualAmount,PlannedAmount,Currency,FiscalPeriod",
            "$expand": "to_CostCenter,to_CostElement",
            "$orderby": "CostCenter,CostElement"
        }
        
        async with self._session.get(endpoint, params=params) as response:
            response.raise_for_status()
            result = await response.json()
            
        # Group by cost center
        cost_centers = {}
        for item in result.get("value", []):
            cc = item.get("CostCenter")
            if cc not in cost_centers:
                cost_centers[cc] = {
                    "cost_center": cc,
                    "name": item.get("CostCenterName"),
                    "actual_total": 0,
                    "plan_total": 0,
                    "currency": item.get("Currency", "EUR"),
                    "cost_elements": []
                }
            
            cost_centers[cc]["actual_total"] += float(item.get("ActualAmount", 0))
            cost_centers[cc]["plan_total"] += float(item.get("PlannedAmount", 0))
            cost_centers[cc]["cost_elements"].append({
                "element": item.get("CostElement"),
                "actual": float(item.get("ActualAmount", 0)),
                "plan": float(item.get("PlannedAmount", 0))
            })
        
        return {
            "cost_center_type": "actual_vs_plan",
            "items": list(cost_centers.values())
        }
    
    async def _get_profit_center_data(self, filter_str: str) -> Dict[str, Any]:
        """Get profit center data"""
        endpoint = f"{self.FINANCIAL_SERVICES['profit_center']}/ProfitCenterActualPlan"
        
        params = {
            "$filter": filter_str,
            "$select": "ProfitCenter,ProfitCenterName,GLAccount,ActualAmount,PlannedAmount,Currency",
            "$orderby": "ProfitCenter,GLAccount"
        }
        
        async with self._session.get(endpoint, params=params) as response:
            response.raise_for_status()
            result = await response.json()
            
        # Group by profit center
        profit_centers = {}
        for item in result.get("value", []):
            pc = item.get("ProfitCenter")
            if pc not in profit_centers:
                profit_centers[pc] = {
                    "profit_center": pc,
                    "name": item.get("ProfitCenterName"),
                    "revenue": 0,
                    "costs": 0,
                    "profit": 0,
                    "currency": item.get("Currency", "EUR"),
                    "accounts": []
                }
            
            amount = float(item.get("ActualAmount", 0))
            gl_account = item.get("GLAccount", "")
            
            # Simple revenue/cost classification based on GL account
            if gl_account.startswith(("4", "8")):  # Revenue accounts
                profit_centers[pc]["revenue"] += amount
            else:  # Cost accounts
                profit_centers[pc]["costs"] += amount
            
            profit_centers[pc]["profit"] = profit_centers[pc]["revenue"] - profit_centers[pc]["costs"]
            profit_centers[pc]["accounts"].append({
                "gl_account": gl_account,
                "actual": amount,
                "plan": float(item.get("PlannedAmount", 0))
            })
        
        return {
            "profit_center_type": "profitability_analysis",
            "items": list(profit_centers.values())
        }
    
    async def _get_gl_account_data(self, filter_str: str) -> Dict[str, Any]:
        """Get GL account line items"""
        endpoint = f"{self.FINANCIAL_SERVICES['gl_account']}/GLAccountLineItem"
        
        params = {
            "$filter": filter_str,
            "$select": "GLAccount,GLAccountName,PostingDate,AmountInCompanyCodeCurrency,Currency,DocumentNumber,DocumentType,Reference",
            "$orderby": "PostingDate desc",
            "$top": 1000  # Limit for performance
        }
        
        async with self._session.get(endpoint, params=params) as response:
            response.raise_for_status()
            result = await response.json()
            
        return {
            "line_item_type": "gl_transactions",
            "items": result.get("value", [])
        }
    
    async def _get_journal_entries(self, filter_str: str) -> Dict[str, Any]:
        """Get journal entries"""
        endpoint = f"{self.FINANCIAL_SERVICES['journal_entry']}/JournalEntry"
        
        params = {
            "$filter": filter_str,
            "$select": "AccountingDocument,PostingDate,DocumentDate,Reference,HeaderText,CreatedByUser",
            "$expand": "to_JournalEntryItem",
            "$orderby": "PostingDate desc",
            "$top": 500
        }
        
        async with self._session.get(endpoint, params=params) as response:
            response.raise_for_status()
            result = await response.json()
            
        # Process journal entries
        entries = []
        for je in result.get("value", []):
            total_debit = sum(float(item.get("AmountInCompanyCodeCurrency", 0)) 
                            for item in je.get("to_JournalEntryItem", []) 
                            if item.get("DebitCreditCode") == "D")
            total_credit = sum(float(item.get("AmountInCompanyCodeCurrency", 0)) 
                             for item in je.get("to_JournalEntryItem", []) 
                             if item.get("DebitCreditCode") == "C")
            
            entries.append({
                "document_number": je.get("AccountingDocument"),
                "posting_date": je.get("PostingDate"),
                "reference": je.get("Reference"),
                "header_text": je.get("HeaderText"),
                "total_debit": total_debit,
                "total_credit": total_credit,
                "is_balanced": abs(total_debit - total_credit) < 0.01,
                "line_items": je.get("to_JournalEntryItem", [])
            })
        
        return {
            "journal_entry_type": "posted_entries",
            "items": entries
        }
    
    async def _get_financial_plan_data(self, filter_str: str) -> Dict[str, Any]:
        """Get financial planning data"""
        # Simplified - would query actual planning tables
        return {
            "plan_type": "financial_forecast",
            "items": []
        }
    
    async def _get_budget_data(self, filter_str: str) -> Dict[str, Any]:
        """Get budget data"""
        # Simplified - would query budget tables
        return {
            "budget_type": "approved_budget",
            "items": []
        }
    
    async def _get_analytics_data(self, company_code: str, fiscal_year: int) -> Dict[str, Any]:
        """Get financial analytics data"""
        # Use SAP Graph for unified analytics
        analytics = await self.sap_graph.get_financial_statements(company_code, fiscal_year)
        
        return {
            "analytics_type": "financial_kpis",
            "items": analytics
        }
    
    @track_performance("s4hana_send_to_a2a")
    async def send_to_a2a_agent(
        self,
        data_product: S4HANADataProduct,
        target_agent_id: Optional[int] = None,
        workflow_id: Optional[str] = None,
        priority: int = 5
    ) -> str:
        """
        Send S/4HANA data to A2A agent for processing
        """
        try:
            # Convert to A2A data product
            a2a_product = data_product.to_a2a_data_product()
            
            # Determine target agent based on data type
            if target_agent_id is None:
                target_agent_id = self.AGENT_ROUTING.get(data_product.type, 0)
            
            # Create agent message
            message = AgentMessage(
                id=f"s4h_msg_{data_product.id}",
                from_agent=f"s4hana_integration",
                to_agent=f"agent_{target_agent_id}",
                message_type="data_product",
                payload={
                    "data_product": a2a_product.dict(),
                    "processing_instructions": {
                        "data_type": data_product.type.value,
                        "validation_required": data_product.type in [
                            S4HANADataType.JOURNAL_ENTRY,
                            S4HANADataType.REGULATORY_REPORT
                        ],
                        "workflow_id": workflow_id
                    }
                },
                priority=priority,
                timestamp=datetime.utcnow()
            )
            
            # Send to message queue
            await self.message_queue.publish(
                f"agent_{target_agent_id}_queue",
                message.dict()
            )
            
            logger.info(f"Sent S/4HANA data {data_product.id} to Agent {target_agent_id}")
            metrics.increment("s4hana.a2a.messages.sent", tags={"agent": target_agent_id})
            
            return message.id
            
        except Exception as e:
            logger.error(f"Failed to send data to A2A agent: {e}")
            metrics.increment("s4hana.a2a.messages.failed")
            raise
    
    async def create_financial_workflow(
        self,
        workflow_type: str,
        company_code: str,
        fiscal_year: int,
        fiscal_period: Optional[int] = None,
        agents: Optional[List[int]] = None
    ) -> WorkflowData:
        """
        Create a financial data processing workflow
        """
        # Default agent pipeline for financial workflows
        if agents is None:
            agents = [0, 1, 2, 4, 5, 6]  # Standard financial validation pipeline
        
        workflow = WorkflowData(
            id=f"s4h_wf_{workflow_type}_{company_code}_{fiscal_year}",
            name=f"S/4HANA {workflow_type} Processing",
            description=f"Process {workflow_type} data for {company_code} FY{fiscal_year}",
            agents=agents,
            config={
                "source": "S4HANA",
                "workflow_type": workflow_type,
                "company_code": company_code,
                "fiscal_year": fiscal_year,
                "fiscal_period": fiscal_period,
                "auto_validation": True,
                "blockchain_logging": True
            },
            status="pending",
            created_at=datetime.utcnow()
        )
        
        # Register workflow with orchestrator (Agent 15)
        await self.send_to_a2a_agent(
            S4HANADataProduct(
                id=workflow.id,
                type=S4HANADataType.FINANCIAL_STATEMENT,
                company_code=company_code,
                fiscal_year=fiscal_year,
                fiscal_period=fiscal_period,
                data={"workflow": workflow.dict()},
                metadata={"workflow_registration": True}
            ),
            target_agent_id=15,  # Orchestrator
            priority=10  # High priority
        )
        
        return workflow
    
    async def sync_master_data_batch(
        self,
        master_data_types: List[str],
        company_code: str,
        agent_id: int = 8  # Data Manager Agent
    ) -> Dict[str, int]:
        """
        Batch sync master data to A2A Data Manager
        """
        results = {}
        
        for data_type in master_data_types:
            try:
                # Extract master data based on type
                if data_type == "cost_centers":
                    data = await self._get_cost_center_data(f"CompanyCode eq '{company_code}'")
                elif data_type == "profit_centers":
                    data = await self._get_profit_center_data(f"CompanyCode eq '{company_code}'")
                elif data_type == "gl_accounts":
                    # Get GL account master data
                    data = {"items": []}  # Simplified
                else:
                    continue
                
                # Create data product
                product = S4HANADataProduct(
                    id=f"master_{data_type}_{company_code}",
                    type=S4HANADataType.ANALYTICS,
                    company_code=company_code,
                    fiscal_year=datetime.now().year,
                    fiscal_period=None,
                    data=data,
                    metadata={
                        "master_data_type": data_type,
                        "sync_timestamp": datetime.utcnow().isoformat()
                    }
                )
                
                # Send to Data Manager
                await self.send_to_a2a_agent(product, target_agent_id=agent_id)
                results[data_type] = len(data.get("items", []))
                
            except Exception as e:
                logger.error(f"Failed to sync {data_type}: {e}")
                results[data_type] = -1
        
        return results
    
    async def monitor_workflow_status(
        self,
        workflow_id: str,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Monitor S/4HANA workflow execution in A2A Network
        """
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).seconds < timeout:
            try:
                # Query workflow status from Control Tower
                # This would integrate with the actual Control Tower monitoring
                status = {
                    "workflow_id": workflow_id,
                    "status": "processing",
                    "progress": 75,
                    "agents_completed": [0, 1, 2, 4],
                    "current_agent": 5,
                    "estimated_completion": "2 minutes"
                }
                
                if status["status"] in ["completed", "failed"]:
                    return status
                
                await asyncio.sleep(5)  # Poll every 5 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring workflow {workflow_id}: {e}")
        
        return {"workflow_id": workflow_id, "status": "timeout"}