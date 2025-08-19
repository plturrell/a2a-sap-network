"""
Financial Domain Preprocessing Pipeline for Enhanced Embeddings
Implements domain-specific normalization and enrichment for financial entities
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FinancialContext:
    """Financial context information for embedding enhancement"""
    normalized_terminology: Dict[str, str]
    regulatory_classification: Dict[str, Any]
    business_hierarchy: Dict[str, str]
    risk_indicators: List[str]
    compliance_flags: List[str]
    domain_synonyms: List[str]


class FinancialDomainNormalizer:
    """
    Normalize financial entities using domain-specific knowledge
    """
    
    def __init__(self):
        self.financial_taxonomies = {
            'account_types': {
                'assets': ['cash', 'receivables', 'inventory', 'investments', 'fixed_assets'],
                'liabilities': ['payables', 'debt', 'provisions', 'derivatives', 'accruals'],
                'equity': ['capital', 'retained_earnings', 'reserves', 'share_premium'],
                'revenue': ['interest_income', 'fee_income', 'trading_income', 'commission'],
                'expenses': ['interest_expense', 'operating_expense', 'provisions', 'depreciation']
            },
            'regulatory_frameworks': {
                'ifrs': ['ifrs9', 'ifrs15', 'ifrs16', 'ifrs17', 'ias36', 'ias39'],
                'basel': ['basel_iii', 'basel_iv', 'crd_iv', 'crr', 'lcr', 'nsfr'],
                'gaap': ['us_gaap', 'local_gaap', 'fasb', 'asc'],
                'regulatory': ['mifid', 'emir', 'gdpr', 'sox', 'coso', 'fatca']
            },
            'business_lines': {
                'retail_banking': ['deposits', 'loans', 'mortgages', 'cards', 'personal_banking'],
                'corporate_banking': ['commercial_lending', 'trade_finance', 'cash_management'],
                'investment_banking': ['capital_markets', 'advisory', 'trading', 'underwriting'],
                'asset_management': ['fund_management', 'custody', 'administration', 'advisory']
            }
        }
        
        # Financial synonym mappings for better semantic matching
        self.synonym_mappings = {
            'gl_account': ['general_ledger', 'account_code', 'chart_of_accounts', 'coa'],
            'book': ['legal_entity', 'consolidation_unit', 'reporting_unit', 'subsidiary'],
            'measure': ['kpi', 'metric', 'performance_indicator', 'financial_ratio', 'benchmark'],
            'product': ['instrument', 'financial_product', 'security_type', 'offering'],
            'location': ['jurisdiction', 'geography', 'regulatory_region', 'domicile']
        }
        
        # Risk classification patterns
        self.risk_patterns = {
            'market_risk': ['trading', 'fx', 'interest_rate', 'equity', 'commodity'],
            'credit_risk': ['lending', 'exposure', 'default', 'counterparty', 'concentration'],
            'operational_risk': ['process', 'system', 'fraud', 'compliance', 'reputation'],
            'liquidity_risk': ['funding', 'market_liquidity', 'maturity_mismatch', 'stress']
        }
    
    def normalize_entity(self, entity_data: Dict[str, Any], entity_type: str) -> FinancialContext:
        """
        Apply comprehensive domain-specific normalization to financial entity
        """
        # 1. Standardize financial terminology
        normalized_terms = self._standardize_terminology(entity_data, entity_type)
        
        # 2. Extract and classify regulatory context
        regulatory_context = self._classify_regulatory_context(entity_data, entity_type)
        
        # 3. Map to business hierarchy and taxonomy
        business_hierarchy = self._map_business_hierarchy(entity_data, entity_type)
        
        # 4. Identify risk indicators
        risk_indicators = self._identify_risk_indicators(entity_data, entity_type)
        
        # 5. Extract compliance flags
        compliance_flags = self._extract_compliance_flags(entity_data, entity_type)
        
        # 6. Generate domain-specific synonyms
        domain_synonyms = self._generate_domain_synonyms(entity_data, entity_type)
        
        return FinancialContext(
            normalized_terminology=normalized_terms,
            regulatory_classification=regulatory_context,
            business_hierarchy=business_hierarchy,
            risk_indicators=risk_indicators,
            compliance_flags=compliance_flags,
            domain_synonyms=domain_synonyms
        )
    
    def _standardize_terminology(self, entity_data: Dict[str, Any], entity_type: str) -> Dict[str, str]:
        """Standardize financial terminology for consistent embeddings"""
        standardized = {}
        entity_str = json.dumps(entity_data, default=str).lower()
        
        # Standardize account terminology
        if entity_type == 'account':
            # Map common account variations to standard terms
            account_mappings = {
                'cash_and_cash_equivalents': ['cash', 'cash_equiv', 'liquid_assets'],
                'accounts_receivable': ['receivables', 'ar', 'trade_receivables'],
                'accounts_payable': ['payables', 'ap', 'trade_payables'],
                'inventory': ['stock', 'goods', 'merchandise'],
                'fixed_assets': ['ppe', 'property_plant_equipment', 'tangible_assets']
            }
            
            for standard_term, variations in account_mappings.items():
                for variation in variations:
                    if variation in entity_str:
                        standardized[variation] = standard_term
        
        # Standardize product terminology
        elif entity_type == 'product':
            product_mappings = {
                'debt_securities': ['bonds', 'notes', 'debentures'],
                'equity_securities': ['stocks', 'shares', 'equity_instruments'],
                'derivatives': ['swaps', 'options', 'futures', 'forwards'],
                'structured_products': ['cds', 'abs', 'mbs', 'cmo']
            }
            
            for standard_term, variations in product_mappings.items():
                for variation in variations:
                    if variation in entity_str:
                        standardized[variation] = standard_term
        
        return standardized
    
    def _classify_regulatory_context(self, entity_data: Dict[str, Any], entity_type: str) -> Dict[str, Any]:
        """Classify regulatory context and applicable frameworks"""
        context = {
            'primary_framework': 'general',
            'applicable_regulations': [],
            'regulatory_scope': 'domestic',
            'classification_confidence': 0.5
        }
        
        entity_str = json.dumps(entity_data, default=str).lower()
        
        # Detect regulatory frameworks
        framework_scores = {}
        for framework, keywords in self.financial_taxonomies['regulatory_frameworks'].items():
            score = sum(1 for keyword in keywords if keyword in entity_str)
            if score > 0:
                framework_scores[framework] = score
                context['applicable_regulations'].extend(keywords)
        
        # Set primary framework
        if framework_scores:
            context['primary_framework'] = max(framework_scores.items(), key=lambda x: x[1])[0]
            context['classification_confidence'] = min(0.9, 0.3 + max(framework_scores.values()) * 0.2)
        
        # Determine regulatory scope
        international_indicators = ['cross_border', 'multinational', 'global', 'offshore']
        if any(indicator in entity_str for indicator in international_indicators):
            context['regulatory_scope'] = 'international'
        
        return context
    
    def _map_business_hierarchy(self, entity_data: Dict[str, Any], entity_type: str) -> Dict[str, str]:
        """Map entity to business hierarchy and operational structure"""
        hierarchy = {
            'business_line': 'general',
            'functional_area': entity_type,
            'operational_level': 'tactical',
            'reporting_segment': 'unspecified'
        }
        
        entity_str = json.dumps(entity_data, default=str).lower()
        
        # Map to business lines
        for business_line, keywords in self.financial_taxonomies['business_lines'].items():
            if any(keyword in entity_str for keyword in keywords):
                hierarchy['business_line'] = business_line
                break
        
        # Determine operational level
        strategic_indicators = ['capital', 'strategy', 'governance', 'board']
        operational_indicators = ['daily', 'transaction', 'processing', 'execution']
        
        if any(indicator in entity_str for indicator in strategic_indicators):
            hierarchy['operational_level'] = 'strategic'
        elif any(indicator in entity_str for indicator in operational_indicators):
            hierarchy['operational_level'] = 'operational'
        
        return hierarchy
    
    def _identify_risk_indicators(self, entity_data: Dict[str, Any], entity_type: str) -> List[str]:
        """Identify risk indicators for risk-aware embeddings"""
        risk_indicators = []
        entity_str = json.dumps(entity_data, default=str).lower()
        
        # Check for risk pattern matches
        for risk_type, patterns in self.risk_patterns.items():
            if any(pattern in entity_str for pattern in patterns):
                risk_indicators.append(risk_type)
        
        # Entity-specific risk assessment
        if entity_type == 'account':
            # High-value accounts
            balance = entity_data.get('balance', 0)
            if isinstance(balance, (int, float)) and balance > 10000000:
                risk_indicators.append('high_value_exposure')
            
            # Foreign currency exposure
            currency = entity_data.get('currency', '').lower()
            if currency not in ['usd', 'eur', 'gbp']:
                risk_indicators.append('foreign_exchange_risk')
                
            # Account type risks
            account_type = entity_data.get('type', '').lower()
            if any(risk_type in account_type for risk_type in ['trading', 'investment', 'derivative']):
                risk_indicators.append('market_risk_exposure')
        
        elif entity_type == 'product':
            # Complex financial products
            product_category = entity_data.get('category', '').lower()
            if any(complex_type in product_category for complex_type in ['structured', 'derivative', 'exotic']):
                risk_indicators.append('complex_instrument_risk')
        
        return list(set(risk_indicators))
    
    def _extract_compliance_flags(self, entity_data: Dict[str, Any], entity_type: str) -> List[str]:
        """Extract compliance and regulatory flags"""
        flags = []
        entity_str = json.dumps(entity_data, default=str).lower()
        
        # Regulatory compliance detection
        compliance_keywords = {
            'sox_compliance': ['sox', 'sarbanes_oxley', 'internal_controls'],
            'mifid_applicable': ['mifid', 'investor_protection', 'best_execution'],
            'basel_requirement': ['basel', 'capital_adequacy', 'risk_weighted'],
            'ifrs_reporting': ['ifrs', 'international_standards', 'fair_value'],
            'gdpr_sensitive': ['personal_data', 'privacy', 'gdpr'],
            'aml_monitoring': ['anti_money_laundering', 'suspicious_activity', 'kyc']
        }
        
        for flag, keywords in compliance_keywords.items():
            if any(keyword in entity_str for keyword in keywords):
                flags.append(flag)
        
        # Entity-specific compliance rules
        if entity_type == 'account':
            account_type = entity_data.get('type', '').lower()
            if account_type in ['trust', 'fiduciary', 'escrow']:
                flags.append('fiduciary_requirements')
            if 'regulatory' in account_type:
                flags.append('regulatory_reporting_required')
                
        return flags
    
    def _generate_domain_synonyms(self, entity_data: Dict[str, Any], entity_type: str) -> List[str]:
        """Generate comprehensive domain-specific synonyms"""
        synonyms = []
        
        # Base synonyms from mapping
        if entity_type in self.synonym_mappings:
            synonyms.extend(self.synonym_mappings[entity_type])
        
        # Extract name variations
        name = entity_data.get('name', '')
        if name:
            synonyms.extend([
                name.lower(),
                name.upper(),
                name.replace(' ', '_'),
                name.replace('_', ' '),
                re.sub(r'[^a-zA-Z0-9]', '', name).lower()
            ])
        
        # Entity-specific synonyms
        if entity_type == 'account':
            account_code = entity_data.get('account_code') or entity_data.get('gl_code')
            if account_code:
                synonyms.append(f"gl_{account_code}")
                synonyms.append(f"account_{account_code}")
                
        elif entity_type == 'product':
            product_id = entity_data.get('product_id') or entity_data.get('instrument_id')
            if product_id:
                synonyms.append(f"instrument_{product_id}")
                synonyms.append(f"product_{product_id}")
        
        return list(set(filter(None, synonyms)))


class ContextualEnrichmentEngine:
    """
    Advanced contextual enrichment for financial entities using domain knowledge
    """
    
    def __init__(self):
        self.context_templates = {
            'account': self._get_account_context_template(),
            'book': self._get_book_context_template(),
            'location': self._get_location_context_template(),
            'measure': self._get_measure_context_template(),
            'product': self._get_product_context_template()
        }
        
        # Financial relationship patterns
        self.relationship_patterns = {
            'hierarchical': ['parent', 'child', 'subsidiary', 'division'],
            'functional': ['feeds_into', 'derives_from', 'controls', 'monitors'],
            'regulatory': ['reports_to', 'complies_with', 'governed_by'],
            'operational': ['processes', 'uses', 'generates', 'consumes']
        }
    
    def enrich_entity_context(self, entity_data: Dict[str, Any], 
                            financial_context: FinancialContext,
                            entity_type: str) -> Dict[str, Any]:
        """
        Generate comprehensive contextual information for enhanced embeddings
        """
        enriched_context = {
            'financial_context': self._create_financial_narrative(entity_data, financial_context, entity_type),
            'business_impact': self._assess_business_impact(entity_data, financial_context, entity_type),
            'regulatory_implications': self._analyze_regulatory_implications(financial_context),
            'risk_assessment': self._create_risk_narrative(financial_context),
            'operational_context': self._build_operational_context(entity_data, entity_type),
            'temporal_factors': self._add_temporal_context(entity_data, entity_type)
        }
        
        return enriched_context
    
    def _create_financial_narrative(self, entity_data: Dict[str, Any], 
                                  financial_context: FinancialContext, 
                                  entity_type: str) -> str:
        """Create comprehensive financial narrative for embedding"""
        narrative_parts = []
        
        # Base entity description
        template = self.context_templates.get(entity_type, "Financial {entity_type}")
        base_desc = template.format(
            **entity_data,
            entity_type=entity_type,
            business_line=financial_context.business_hierarchy.get('business_line', 'general'),
            regulatory_framework=financial_context.regulatory_classification.get('primary_framework', 'general')
        )
        narrative_parts.append(base_desc)
        
        # Add regulatory context
        if financial_context.regulatory_classification['applicable_regulations']:
            regs = ', '.join(financial_context.regulatory_classification['applicable_regulations'][:3])
            narrative_parts.append(f"Subject to {regs} regulatory requirements")
        
        # Add risk context
        if financial_context.risk_indicators:
            risks = ', '.join(financial_context.risk_indicators[:2])
            narrative_parts.append(f"Associated with {risks}")
        
        # Add business hierarchy context
        business_line = financial_context.business_hierarchy.get('business_line', '')
        if business_line and business_line != 'general':
            narrative_parts.append(f"Part of {business_line.replace('_', ' ')} operations")
        
        return '. '.join(narrative_parts)
    
    def _assess_business_impact(self, entity_data: Dict[str, Any], 
                               financial_context: FinancialContext, 
                               entity_type: str) -> Dict[str, Any]:
        """Assess business impact and criticality"""
        impact_assessment = {
            'criticality_level': 'medium',
            'stakeholder_impact': [],
            'process_dependencies': [],
            'revenue_impact': 'indirect'
        }
        
        # Determine criticality based on entity attributes
        if entity_type == 'account':
            balance = entity_data.get('balance', 0)
            if isinstance(balance, (int, float)):
                if balance > 100000000:
                    impact_assessment['criticality_level'] = 'critical'
                elif balance > 10000000:
                    impact_assessment['criticality_level'] = 'high'
        
        # Map business line to stakeholders
        business_line = financial_context.business_hierarchy.get('business_line', '')
        stakeholder_mapping = {
            'retail_banking': ['customers', 'branch_operations', 'compliance'],
            'corporate_banking': ['corporate_clients', 'relationship_managers', 'credit_risk'],
            'investment_banking': ['institutional_clients', 'traders', 'market_risk'],
            'asset_management': ['fund_investors', 'portfolio_managers', 'operations']
        }
        
        if business_line in stakeholder_mapping:
            impact_assessment['stakeholder_impact'] = stakeholder_mapping[business_line]
        
        return impact_assessment
    
    def _analyze_regulatory_implications(self, financial_context: FinancialContext) -> Dict[str, Any]:
        """Analyze regulatory implications and requirements"""
        implications = {
            'compliance_complexity': 'low',
            'reporting_requirements': [],
            'audit_frequency': 'annual',
            'regulatory_scrutiny': 'standard'
        }
        
        # Assess complexity based on applicable regulations
        reg_count = len(financial_context.regulatory_classification['applicable_regulations'])
        if reg_count > 5:
            implications['compliance_complexity'] = 'high'
        elif reg_count > 2:
            implications['compliance_complexity'] = 'medium'
        
        # Map regulations to reporting requirements
        regulation_reporting = {
            'sox': ['internal_controls_testing', 'management_certification'],
            'basel': ['capital_adequacy_reporting', 'risk_weighted_assets'],
            'mifid': ['transaction_reporting', 'best_execution_reporting'],
            'ifrs': ['fair_value_reporting', 'impairment_testing']
        }
        
        for reg in financial_context.regulatory_classification['applicable_regulations']:
            if reg in regulation_reporting:
                implications['reporting_requirements'].extend(regulation_reporting[reg])
        
        return implications
    
    def _create_risk_narrative(self, financial_context: FinancialContext) -> str:
        """Create risk-focused narrative for embedding"""
        if not financial_context.risk_indicators:
            return "Standard risk profile with typical financial controls"
        
        risk_descriptions = {
            'market_risk': 'exposed to market volatility and price fluctuations',
            'credit_risk': 'subject to counterparty default and credit deterioration',
            'operational_risk': 'vulnerable to process failures and operational disruptions',
            'liquidity_risk': 'sensitive to funding availability and market liquidity',
            'high_value_exposure': 'represents significant financial exposure requiring enhanced controls',
            'foreign_exchange_risk': 'affected by currency exchange rate movements',
            'complex_instrument_risk': 'involves complex financial instruments requiring specialized expertise'
        }
        
        narratives = []
        for risk in financial_context.risk_indicators[:3]:  # Top 3 risks
            if risk in risk_descriptions:
                narratives.append(risk_descriptions[risk])
        
        return f"Risk profile: {', '.join(narratives)}"
    
    def _build_operational_context(self, entity_data: Dict[str, Any], entity_type: str) -> Dict[str, Any]:
        """Build operational context for embedding"""
        context = {
            'processing_frequency': 'daily',
            'automation_level': 'medium',
            'system_integration': 'standard',
            'data_sensitivity': 'medium'
        }
        
        # Entity-specific operational patterns
        if entity_type == 'account':
            account_type = entity_data.get('type', '').lower()
            if 'trading' in account_type:
                context['processing_frequency'] = 'real_time'
                context['automation_level'] = 'high'
            elif 'regulatory' in account_type:
                context['data_sensitivity'] = 'high'
                context['system_integration'] = 'specialized'
        
        return context
    
    def _add_temporal_context(self, entity_data: Dict[str, Any], entity_type: str) -> Dict[str, Any]:
        """Add temporal and cyclical context"""
        temporal_context = {
            'reporting_cycle': 'monthly',
            'peak_activity_periods': ['month_end', 'quarter_end'],
            'historical_retention': '7_years',
            'update_frequency': 'daily'
        }
        
        # Regulatory-driven temporal patterns
        if entity_type == 'account':
            if entity_data.get('type', '').lower() in ['regulatory', 'capital']:
                temporal_context['reporting_cycle'] = 'quarterly'
                temporal_context['peak_activity_periods'] = ['quarter_end', 'year_end']
        
        return temporal_context
    
    def _get_account_context_template(self) -> str:
        return """Financial account entity in the {business_line} business line, classified under the {regulatory_framework} regulatory framework. 
        This account represents {type} with operational significance in financial reporting and regulatory compliance."""
    
    def _get_book_context_template(self) -> str:
        return """Legal entity book for {business_line} operations, serving as a consolidation unit for regulatory reporting under {regulatory_framework} requirements."""
    
    def _get_location_context_template(self) -> str:
        return """Geographic location entity supporting {business_line} operations, with regulatory jurisdiction under {regulatory_framework} oversight."""
    
    def _get_measure_context_template(self) -> str:
        return """Financial performance measure for {business_line} business line, used for monitoring and reporting under {regulatory_framework} standards."""
    
    def _get_product_context_template(self) -> str:
        return """Financial product offering within {business_line}, regulated under {regulatory_framework} with specific compliance and risk management requirements."""


class FinancialPromptEngineer:
    """
    Generate domain-optimized prompts for superior embedding quality
    """
    
    def __init__(self):
        self.prompt_strategies = {
            'hierarchical': self._create_hierarchical_prompt,
            'regulatory': self._create_regulatory_prompt,
            'risk_based': self._create_risk_based_prompt,
            'business_process': self._create_business_process_prompt,
            'temporal': self._create_temporal_prompt
        }
        
        self.strategy_weights = {
            'hierarchical': 0.25,
            'regulatory': 0.30,
            'risk_based': 0.25,
            'business_process': 0.15,
            'temporal': 0.05
        }
    
    def engineer_embedding_prompt(self, entity_data: Dict[str, Any], 
                                financial_context: FinancialContext,
                                enriched_context: Dict[str, Any],
                                entity_type: str) -> str:
        """
        Create multi-perspective optimized prompt for financial embedding generation
        """
        # Generate strategy-specific prompts
        strategy_prompts = {}
        for strategy, prompt_func in self.prompt_strategies.items():
            strategy_prompts[strategy] = prompt_func(entity_data, financial_context, enriched_context, entity_type)
        
        # Combine prompts with weighted importance
        final_prompt = self._combine_weighted_prompts(strategy_prompts, entity_type)
        
        # Add domain-specific instructions
        domain_instructions = self._get_financial_domain_instructions(entity_type)
        
        return f"{domain_instructions}\n\n{final_prompt}"
    
    def _create_hierarchical_prompt(self, entity_data: Dict, financial_context: FinancialContext, 
                                  enriched_context: Dict, entity_type: str) -> str:
        """Create hierarchical structure-aware prompt"""
        hierarchy = financial_context.business_hierarchy
        
        if entity_type == 'account':
            return f"""
            Financial Account Hierarchy Context:
            Business Line: {hierarchy.get('business_line', 'General')}
            Functional Area: {hierarchy.get('functional_area', entity_type)}
            Operational Level: {hierarchy.get('operational_level', 'Tactical')}
            Reporting Segment: {hierarchy.get('reporting_segment', 'Unspecified')}
            
            This hierarchical positioning defines the account's role in organizational structure,
            reporting relationships, and operational dependencies within the financial ecosystem.
            """
        
        return f"""
        Entity Hierarchy: {hierarchy.get('business_line', 'General')} > {hierarchy.get('functional_area', entity_type)}
        Operational Level: {hierarchy.get('operational_level', 'Tactical')}
        """
    
    def _create_regulatory_prompt(self, entity_data: Dict, financial_context: FinancialContext,
                                enriched_context: Dict, entity_type: str) -> str:
        """Create regulatory framework-aware prompt"""
        reg_context = financial_context.regulatory_classification
        reg_implications = enriched_context.get('regulatory_implications', {})
        
        prompt = f"""
        Regulatory Classification and Compliance Context:
        Primary Framework: {reg_context.get('primary_framework', 'General').upper()}
        Regulatory Scope: {reg_context.get('regulatory_scope', 'Domestic')}
        Applicable Regulations: {', '.join(reg_context.get('applicable_regulations', [])[:4])}
        Compliance Complexity: {reg_implications.get('compliance_complexity', 'Medium')}
        
        Compliance Flags: {', '.join(financial_context.compliance_flags[:3]) if financial_context.compliance_flags else 'Standard compliance requirements'}
        
        This regulatory context determines reporting obligations, audit requirements,
        and compliance monitoring for this financial entity.
        """
        
        return prompt
    
    def _create_risk_based_prompt(self, entity_data: Dict, financial_context: FinancialContext,
                                enriched_context: Dict, entity_type: str) -> str:
        """Create risk-aware prompt"""
        risk_narrative = enriched_context.get('risk_assessment', 'Standard risk profile')
        
        prompt = f"""
        Risk Profile and Management Context:
        Risk Indicators: {', '.join(financial_context.risk_indicators[:4]) if financial_context.risk_indicators else 'Standard risk profile'}
        Risk Assessment: {risk_narrative}
        
        Business Impact Assessment:
        Criticality Level: {enriched_context.get('business_impact', {}).get('criticality_level', 'Medium')}
        Stakeholder Impact: {', '.join(enriched_context.get('business_impact', {}).get('stakeholder_impact', [])[:3])}
        
        This risk context influences control requirements, monitoring intensity,
        and management attention for this financial entity.
        """
        
        return prompt
    
    def _create_business_process_prompt(self, entity_data: Dict, financial_context: FinancialContext,
                                      enriched_context: Dict, entity_type: str) -> str:
        """Create business process context prompt"""
        financial_narrative = enriched_context.get('financial_context', f'{entity_type} entity')
        operational_context = enriched_context.get('operational_context', {})
        
        prompt = f"""
        Business Process and Operational Context:
        {financial_narrative}
        
        Operational Characteristics:
        Processing Frequency: {operational_context.get('processing_frequency', 'Daily')}
        Automation Level: {operational_context.get('automation_level', 'Medium')}
        System Integration: {operational_context.get('system_integration', 'Standard')}
        Data Sensitivity: {operational_context.get('data_sensitivity', 'Medium')}
        
        This operational context defines how the entity integrates into business processes
        and workflow dependencies.
        """
        
        return prompt
    
    def _create_temporal_prompt(self, entity_data: Dict, financial_context: FinancialContext,
                              enriched_context: Dict, entity_type: str) -> str:
        """Create temporal and cyclical context prompt"""
        temporal_factors = enriched_context.get('temporal_factors', {})
        
        prompt = f"""
        Temporal and Cyclical Context:
        Reporting Cycle: {temporal_factors.get('reporting_cycle', 'Monthly')}
        Peak Activity: {', '.join(temporal_factors.get('peak_activity_periods', ['Month-end']))}
        Update Frequency: {temporal_factors.get('update_frequency', 'Daily')}
        Retention Period: {temporal_factors.get('historical_retention', '7 years')}
        
        These temporal patterns affect processing priorities, resource allocation,
        and system performance requirements.
        """
        
        return prompt
    
    def _combine_weighted_prompts(self, strategy_prompts: Dict[str, str], entity_type: str) -> str:
        """Combine strategy prompts with appropriate weighting"""
        combined_sections = []
        
        # Order sections by weight (highest first)
        ordered_strategies = sorted(self.strategy_weights.items(), key=lambda x: x[1], reverse=True)
        
        for strategy, weight in ordered_strategies:
            if strategy in strategy_prompts and strategy_prompts[strategy].strip():
                section_header = f"=== {strategy.replace('_', ' ').title()} Perspective (Weight: {weight:.0%}) ==="
                combined_sections.append(f"{section_header}\n{strategy_prompts[strategy]}")
        
        return "\n\n".join(combined_sections)
    
    def _get_financial_domain_instructions(self, entity_type: str) -> str:
        """Get financial domain-specific embedding instructions"""
        base_instructions = """
        FINANCIAL DOMAIN EMBEDDING INSTRUCTIONS:
        This is a financial entity requiring domain-aware semantic representation.
        Key considerations for embedding generation:
        
        1. REGULATORY COMPLIANCE: Emphasize regulatory framework alignment and compliance requirements
        2. RISK CONTEXT: Integrate risk indicators and management implications
        3. BUSINESS HIERARCHY: Preserve organizational structure and reporting relationships
        4. OPERATIONAL INTEGRATION: Maintain business process context and workflow dependencies
        5. TEMPORAL FACTORS: Include cyclical patterns and temporal dependencies
        
        Generate embeddings that capture financial domain semantics, regulatory nuances, 
        and business context for superior similarity matching and retrieval.
        """
        
        entity_specific = {
            'account': "Focus on accounting principles, regulatory reporting, and financial statement integration.",
            'product': "Emphasize instrument characteristics, market dynamics, and product lifecycle management.",
            'measure': "Highlight performance metrics, benchmarking context, and analytical applications.",
            'location': "Include jurisdictional implications, regulatory geography, and operational scope.",
            'book': "Focus on legal entity structure, consolidation rules, and reporting hierarchies."
        }
        
        specific_instruction = entity_specific.get(entity_type, "Apply general financial entity processing guidelines.")
        
        return f"{base_instructions}\n\nENTITY-SPECIFIC: {specific_instruction}"