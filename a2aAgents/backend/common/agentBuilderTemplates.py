"""
Enhanced Agent Builder templates and SDK management for A2A system.
Provides comprehensive agent generation, templating, and SDK management capabilities.
"""
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import jinja2
import yaml

from config.agentConfig import config
from common.errorHandling import with_circuit_breaker, with_retry

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of agents that can be generated."""
    DATA_PROCESSOR = "data_processor"
    VALIDATOR = "validator"
    CALCULATOR = "calculator"
    REASONER = "reasoner"
    INTEGRATOR = "integrator"
    CUSTOM = "custom"


class TemplateType(Enum):
    """Types of templates available."""
    BASIC = "basic"
    ADVANCED = "advanced"
    ENTERPRISE = "enterprise"
    MICROSERVICE = "microservice"


class SDKComponent(Enum):
    """SDK components that can be included."""
    TRUST_SYSTEM = "trust_system"
    ERROR_HANDLING = "error_handling"
    MONITORING = "monitoring"
    PERFORMANCE = "performance"
    BLOCKCHAIN = "blockchain"
    VECTOR_OPS = "vector_ops"
    DATA_LIFECYCLE = "data_lifecycle"
    SCHEMA_VERSIONING = "schema_versioning"


@dataclass
class AgentTemplate:
    """Agent template definition."""
    template_id: str
    name: str
    description: str
    agent_type: AgentType
    template_type: TemplateType
    template_path: str
    required_components: List[SDKComponent]
    optional_components: List[SDKComponent]
    configuration_schema: Dict[str, Any]
    example_config: Dict[str, Any]
    created_at: datetime
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)


@dataclass
class AgentConfiguration:
    """Agent configuration for generation."""
    agent_id: str
    agent_name: str
    agent_description: str
    agent_type: AgentType
    template_id: str
    sdk_components: List[SDKComponent]
    custom_config: Dict[str, Any]
    output_path: str
    enable_tests: bool = True
    enable_docs: bool = True


@dataclass
class GenerationResult:
    """Result of agent generation."""
    success: bool
    agent_path: str
    generated_files: List[str]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    generation_time: float = 0.0


class EnhancedAgentBuilder:
    """
    Enhanced agent builder with templates and SDK management.
    """
    
    def __init__(
        self,
        templates_path: str,
        output_base_path: str,
        sdk_path: str
    ):
        """
        Initialize enhanced agent builder.
        
        Args:
            templates_path: Path to agent templates
            output_base_path: Base path for generated agents
            sdk_path: Path to SDK components
        """
        self.templates_path = Path(templates_path)
        self.output_base_path = Path(output_base_path)
        self.sdk_path = Path(sdk_path)
        
        # Template registry
        self.templates = {}  # template_id -> AgentTemplate
        
        # Jinja2 environment
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.templates_path)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # SDK component mappings
        self.sdk_components = {
            SDKComponent.TRUST_SYSTEM: {
                'imports': ['from common.standardTrustMixin import TrustSystemMixin'],
                'mixins': ['TrustSystemMixin'],
                'dependencies': ['trustSystem.trustIntegration']
            },
            SDKComponent.ERROR_HANDLING: {
                'imports': ['from common.errorHandling import with_circuit_breaker, with_retry, ErrorContext'],
                'mixins': [],
                'dependencies': ['common.errorHandling']
            },
            SDKComponent.MONITORING: {
                'imports': ['from monitoring.prometheusConfig import create_agent_metrics'],
                'mixins': [],
                'dependencies': ['prometheus_client']
            },
            SDKComponent.PERFORMANCE: {
                'imports': ['from app.a2a.core.performanceOptimizer import PerformanceOptimizationMixin'],
                'mixins': ['PerformanceOptimizationMixin'],
                'dependencies': []
            },
            SDKComponent.BLOCKCHAIN: {
                'imports': ['from trustSystem.trustIntegration import get_trust_system'],
                'mixins': [],
                'dependencies': ['web3', 'eth_account']
            },
            SDKComponent.VECTOR_OPS: {
                'imports': ['from common.hanaVectorEngine import HANAVectorEngine'],
                'mixins': [],
                'dependencies': ['numpy', 'hdbcli']
            },
            SDKComponent.DATA_LIFECYCLE: {
                'imports': ['from common.dataLifecycle import DataLifecycleManager'],
                'mixins': [],
                'dependencies': ['pathlib']
            },
            SDKComponent.SCHEMA_VERSIONING: {
                'imports': ['from common.schemaVersioning import SchemaVersioningManager'],
                'mixins': [],
                'dependencies': ['semver']
            }
        }
        
        # Initialize
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize built-in templates."""
        # Create templates directory if it doesn't exist
        self.templates_path.mkdir(parents=True, exist_ok=True)
        
        # Load built-in templates
        self._create_builtin_templates()
        self._load_existing_templates()
    
    def _create_builtin_templates(self):
        """Create built-in agent templates."""
        # Basic Data Processor Template
        basic_data_processor = AgentTemplate(
            template_id="basic_data_processor",
            name="Basic Data Processor",
            description="Simple data processing agent with standard A2A capabilities",
            agent_type=AgentType.DATA_PROCESSOR,
            template_type=TemplateType.BASIC,
            template_path="basic_data_processor.py.j2",
            required_components=[SDKComponent.TRUST_SYSTEM, SDKComponent.ERROR_HANDLING],
            optional_components=[SDKComponent.MONITORING, SDKComponent.PERFORMANCE],
            configuration_schema={
                "type": "object",
                "properties": {
                    "processing_batch_size": {"type": "integer", "default": 100},
                    "enable_caching": {"type": "boolean", "default": True},
                    "cache_ttl": {"type": "integer", "default": 300}
                }
            },
            example_config={
                "processing_batch_size": 100,
                "enable_caching": True,
                "cache_ttl": 300
            },
            created_at=datetime.now()
        )
        
        # Advanced Validator Template
        advanced_validator = AgentTemplate(
            template_id="advanced_validator",
            name="Advanced Validator",
            description="Comprehensive validation agent with multiple validation strategies",
            agent_type=AgentType.VALIDATOR,
            template_type=TemplateType.ADVANCED,
            template_path="advanced_validator.py.j2",
            required_components=[
                SDKComponent.TRUST_SYSTEM,
                SDKComponent.ERROR_HANDLING,
                SDKComponent.MONITORING
            ],
            optional_components=[
                SDKComponent.BLOCKCHAIN,
                SDKComponent.SCHEMA_VERSIONING
            ],
            configuration_schema={
                "type": "object",
                "properties": {
                    "validation_strategies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": ["structural", "semantic", "business"]
                    },
                    "strict_mode": {"type": "boolean", "default": False},
                    "validation_timeout": {"type": "integer", "default": 30}
                }
            },
            example_config={
                "validation_strategies": ["structural", "semantic", "business"],
                "strict_mode": False,
                "validation_timeout": 30
            },
            created_at=datetime.now()
        )
        
        # Enterprise Calculator Template
        enterprise_calculator = AgentTemplate(
            template_id="enterprise_calculator",
            name="Enterprise Calculator",
            description="High-performance calculation agent with self-healing capabilities",
            agent_type=AgentType.CALCULATOR,
            template_type=TemplateType.ENTERPRISE,
            template_path="enterprise_calculator.py.j2",
            required_components=[
                SDKComponent.TRUST_SYSTEM,
                SDKComponent.ERROR_HANDLING,
                SDKComponent.MONITORING,
                SDKComponent.PERFORMANCE
            ],
            optional_components=[
                SDKComponent.BLOCKCHAIN,
                SDKComponent.VECTOR_OPS
            ],
            configuration_schema={
                "type": "object",
                "properties": {
                    "calculation_precision": {"type": "string", "default": "high"},
                    "enable_self_healing": {"type": "boolean", "default": True},
                    "parallel_processing": {"type": "boolean", "default": True},
                    "max_workers": {"type": "integer", "default": 4}
                }
            },
            example_config={
                "calculation_precision": "high",
                "enable_self_healing": True,
                "parallel_processing": True,
                "max_workers": 4
            },
            created_at=datetime.now()
        )
        
        # Store templates
        for template in [basic_data_processor, advanced_validator, enterprise_calculator]:
            self.templates[template.template_id] = template
            self._create_template_file(template)
    
    def _create_template_file(self, template: AgentTemplate):
        """Create template file on disk."""
        template_file = self.templates_path / template.template_path
        
        if template_file.exists():
            return  # Don't overwrite existing templates
        
        # Generate template content based on type
        template_content = self._generate_template_content(template)
        
        with open(template_file, 'w') as f:
            f.write(template_content)
        
        logger.debug(f"Created template file: {template_file}")
    
    def _generate_template_content(self, template: AgentTemplate) -> str:
        """Generate template file content."""
        if template.agent_type == AgentType.DATA_PROCESSOR:
            return self._generate_data_processor_template()
        elif template.agent_type == AgentType.VALIDATOR:
            return self._generate_validator_template()
        elif template.agent_type == AgentType.CALCULATOR:
            return self._generate_calculator_template()
        else:
            return self._generate_basic_template()
    
    def _generate_data_processor_template(self) -> str:
        """Generate data processor template."""
        return '''"""
{{ agent_description }}
Generated by A2A Enhanced Agent Builder
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# A2A SDK imports
from app.a2a.sdk import (
    A2AAgentBase, a2a_handler, a2a_skill, a2a_task,
    A2AMessage, MessageRole, create_agent_id
)

{% for import_stmt in sdk_imports %}
{{ import_stmt }}
{% endfor %}

logger = logging.getLogger(__name__)


class {{ agent_class_name }}(A2AAgentBase{% for mixin in sdk_mixins %}, {{ mixin }}{% endfor %}):
    """
    {{ agent_description }}
    """
    
    def __init__(self, base_url: str, **kwargs):
        """Initialize {{ agent_name }}."""
        super().__init__(
            agent_id="{{ agent_id }}",
            name="{{ agent_name }}",
            description="{{ agent_description }}",
            version="1.0.0",
            base_url=base_url
        )
        
        # Configuration
        self.config = kwargs.get('config', {})
        self.processing_batch_size = self.config.get('processing_batch_size', 100)
        self.enable_caching = self.config.get('enable_caching', True)
        self.cache_ttl = self.config.get('cache_ttl', 300)
        
        # Initialize SDK components
        {% if 'TrustSystemMixin' in sdk_mixins %}
        asyncio.create_task(self.initialize_trust_system())
        {% endif %}
        
        {% if monitoring_enabled %}
        self.metrics = create_agent_metrics("{{ agent_id }}", "data_processor")
        {% endif %}
        
        logger.info(f"Initialized {{ agent_name }}")
    
    @a2a_skill("data_processing")
    async def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming data."""
        start_time = datetime.now()
        
        try:
            {% if error_handling_enabled %}
            with ErrorContext("data_processing"):
            {% endif %}
                # Process data in batches
                results = []
                data_items = data.get('items', [])
                
                for i in range(0, len(data_items), self.processing_batch_size):
                    batch = data_items[i:i + self.processing_batch_size]
                    batch_result = await self._process_batch(batch)
                    results.extend(batch_result)
                
                {% if monitoring_enabled %}
                processing_time = (datetime.now() - start_time).total_seconds()
                self.metrics.record_task("data_processing", "success", processing_time)
                self.metrics.record_data_processed("processing", len(data_items))
                {% endif %}
                
                return {
                    'success': True,
                    'processed_items': len(results),
                    'results': results,
                    'processing_time': processing_time
                }
        
        except Exception as e:
            {% if monitoring_enabled %}
            self.metrics.record_task("data_processing", "failure")
            {% endif %}
            logger.error(f"Data processing failed: {e}")
            raise
    
    async def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of data items."""
        results = []
        
        for item in batch:
            # Apply processing logic
            processed_item = await self._process_item(item)
            results.append(processed_item)
        
        return results
    
    async def _process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual data item."""
        # Implement specific processing logic here
        processed = {
            'id': item.get('id'),
            'processed_at': datetime.now().isoformat(),
            'data': item.get('data'),
            'metadata': {
                'processor': "{{ agent_id }}",
                'version': "1.0.0"
            }
        }
        
        return processed


if __name__ == "__main__":
    import sys
    
    agent = {{ agent_class_name }}(base_url=os.getenv("A2A_SERVICE_URL"))
    # Add startup logic here
'''
    
    def _generate_validator_template(self) -> str:
        """Generate validator template."""
        return '''"""
{{ agent_description }}
Generated by A2A Enhanced Agent Builder
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# A2A SDK imports
from app.a2a.sdk import (
    A2AAgentBase, a2a_handler, a2a_skill, a2a_task,
    A2AMessage, MessageRole, create_agent_id
)

{% for import_stmt in sdk_imports %}
{{ import_stmt }}
{% endfor %}

logger = logging.getLogger(__name__)


class {{ agent_class_name }}(A2AAgentBase{% for mixin in sdk_mixins %}, {{ mixin }}{% endfor %}):
    """
    {{ agent_description }}
    """
    
    def __init__(self, base_url: str, **kwargs):
        """Initialize {{ agent_name }}."""
        super().__init__(
            agent_id="{{ agent_id }}",
            name="{{ agent_name }}",
            description="{{ agent_description }}",
            version="1.0.0",
            base_url=base_url
        )
        
        # Configuration
        self.config = kwargs.get('config', {})
        self.validation_strategies = self.config.get('validation_strategies', ['structural', 'semantic'])
        self.strict_mode = self.config.get('strict_mode', False)
        self.validation_timeout = self.config.get('validation_timeout', 30)
        
        # Validation rules
        self.validation_rules = {}
        self._load_validation_rules()
        
        {% if monitoring_enabled %}
        self.metrics = create_agent_metrics("{{ agent_id }}", "validator")
        {% endif %}
        
        logger.info(f"Initialized {{ agent_name }} with strategies: {self.validation_strategies}")
    
    @a2a_skill("validation")
    async def validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data using configured strategies."""
        validation_results = {
            'is_valid': True,
            'validation_id': f"val_{datetime.now().timestamp()}",
            'strategy_results': {},
            'issues': [],
            'warnings': []
        }
        
        try:
            for strategy in self.validation_strategies:
                {% if error_handling_enabled %}
                with ErrorContext(f"validation_{strategy}"):
                {% endif %}
                    strategy_result = await self._execute_validation_strategy(strategy, data)
                    validation_results['strategy_results'][strategy] = strategy_result
                    
                    if not strategy_result['valid']:
                        validation_results['is_valid'] = False
                        validation_results['issues'].extend(strategy_result.get('issues', []))
                    
                    validation_results['warnings'].extend(strategy_result.get('warnings', []))
            
            {% if monitoring_enabled %}
            self.metrics.record_validation("data_validation", "success" if validation_results['is_valid'] else "failure")
            {% endif %}
            
            return validation_results
        
        except Exception as e:
            {% if monitoring_enabled %}
            self.metrics.record_validation("data_validation", "error")
            {% endif %}
            logger.error(f"Validation failed: {e}")
            raise
    
    async def _execute_validation_strategy(self, strategy: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute specific validation strategy."""
        if strategy == "structural":
            return await self._structural_validation(data)
        elif strategy == "semantic":
            return await self._semantic_validation(data)
        elif strategy == "business":
            return await self._business_validation(data)
        else:
            return {'valid': True, 'issues': [], 'warnings': [f"Unknown strategy: {strategy}"]}
    
    async def _structural_validation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform structural validation."""
        issues = []
        warnings = []
        
        # Check required fields
        required_fields = ['id', 'type', 'data']
        for field in required_fields:
            if field not in data:
                issues.append(f"Missing required field: {field}")
        
        # Check data types
        if 'id' in data and not isinstance(data['id'], str):
            issues.append("Field 'id' must be a string")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    async def _semantic_validation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform semantic validation."""
        issues = []
        warnings = []
        
        # Implement semantic validation logic
        # This would typically involve checking business rules, relationships, etc.
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    async def _business_validation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform business rule validation."""
        issues = []
        warnings = []
        
        # Implement business rule validation
        # This would check domain-specific constraints
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    def _load_validation_rules(self):
        """Load validation rules from configuration."""
        # Load standard validation rules
        self.validation_rules = {
            'required_fields': ['id', 'type', 'data'],
            'field_types': {
                'id': str,
                'type': str,
                'data': dict
            },
            'business_rules': {
                'max_data_size': 1024 * 1024,  # 1MB max
                'allowed_types': ['document', 'record', 'transaction'],
                'required_metadata': ['created_at', 'version']
            },
            'semantic_rules': {
                'relationship_validation': True,
                'reference_integrity': True,
                'domain_constraints': True
            }
        }


if __name__ == "__main__":
    import sys
    
    agent = {{ agent_class_name }}(base_url=os.getenv("A2A_SERVICE_URL"))
    # Add startup logic here
'''
    
    def _generate_calculator_template(self) -> str:
        """Generate calculator template."""
        return '''"""
{{ agent_description }}
Generated by A2A Enhanced Agent Builder
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import concurrent.futures

# A2A SDK imports
from app.a2a.sdk import (
    A2AAgentBase, a2a_handler, a2a_skill, a2a_task,
    A2AMessage, MessageRole, create_agent_id
)

{% for import_stmt in sdk_imports %}
{{ import_stmt }}
{% endfor %}

logger = logging.getLogger(__name__)


class {{ agent_class_name }}(A2AAgentBase{% for mixin in sdk_mixins %}, {{ mixin }}{% endfor %}):
    """
    {{ agent_description }}
    """
    
    def __init__(self, base_url: str, **kwargs):
        """Initialize {{ agent_name }}."""
        super().__init__(
            agent_id="{{ agent_id }}",
            name="{{ agent_name }}",
            description="{{ agent_description }}",
            version="1.0.0",
            base_url=base_url
        )
        
        # Configuration
        self.config = kwargs.get('config', {})
        self.calculation_precision = self.config.get('calculation_precision', 'high')
        self.enable_self_healing = self.config.get('enable_self_healing', True)
        self.parallel_processing = self.config.get('parallel_processing', True)
        self.max_workers = self.config.get('max_workers', 4)
        
        # Initialize thread pool for parallel processing
        if self.parallel_processing:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        
        {% if self_healing_enabled %}
        from common.selfHealing import SelfHealingCalculator
        self.self_healing = SelfHealingCalculator("{{ agent_id }}")
        {% endif %}
        
        {% if monitoring_enabled %}
        self.metrics = create_agent_metrics("{{ agent_id }}", "calculator")
        {% endif %}
        
        logger.info(f"Initialized {{ agent_name }} with precision: {self.calculation_precision}")
    
    @a2a_skill("calculation")
    async def perform_calculation(self, calculation_request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform mathematical calculations."""
        calculation_id = f"calc_{datetime.now().timestamp()}"
        
        try:
            operation = calculation_request.get('operation')
            operands = calculation_request.get('operands', [])
            
            {% if self_healing_enabled %}
            if self.enable_self_healing:
                result = await self.self_healing.calculate_with_healing(
                    self._execute_calculation,
                    operation,
                    operands
                )
            else:
                result = await self._execute_calculation(operation, operands)
            {% else %}
            result = await self._execute_calculation(operation, operands)
            {% endif %}
            
            {% if monitoring_enabled %}
            self.metrics.record_task("calculation", "success")
            {% endif %}
            
            return {
                'calculation_id': calculation_id,
                'success': True,
                'result': result,
                'operation': operation,
                'operands': operands,
                'precision': self.calculation_precision,
                'calculated_at': datetime.now().isoformat()
            }
        
        except Exception as e:
            {% if monitoring_enabled %}
            self.metrics.record_task("calculation", "failure")
            {% endif %}
            logger.error(f"Calculation {calculation_id} failed: {e}")
            raise
    
    async def _execute_calculation(self, operation: str, operands: List[Union[int, float]]) -> Union[int, float]:
        """Execute the actual calculation."""
        if operation == "add":
            return sum(operands)
        elif operation == "subtract":
            return operands[0] - sum(operands[1:])
        elif operation == "multiply":
            result = 1
            for operand in operands:
                result *= operand
            return result
        elif operation == "divide":
            if len(operands) < 2:
                raise ValueError("Division requires at least 2 operands")
            result = operands[0]
            for operand in operands[1:]:
                if operand == 0:
                    raise ValueError("Division by zero")
                result /= operand
            return result
        elif operation == "power":
            if len(operands) != 2:
                raise ValueError("Power operation requires exactly 2 operands")
            return operands[0] ** operands[1]
        elif operation == "sqrt":
            if len(operands) != 1:
                raise ValueError("Square root requires exactly 1 operand")
            import math
            return math.sqrt(operands[0])
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    @a2a_skill("batch_calculation")
    async def batch_calculate(self, batch_request: Dict[str, Any]) -> Dict[str, Any]:
        """Perform batch calculations."""
        calculations = batch_request.get('calculations', [])
        
        if self.parallel_processing:
            # Process calculations in parallel
            tasks = []
            for calc in calculations:
                task = self.perform_calculation(calc)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Process calculations sequentially
            results = []
            for calc in calculations:
                try:
                    result = await self.perform_calculation(calc)
                    results.append(result)
                except Exception as e:
                    results.append({'error': str(e)})
        
        return {
            'batch_id': f"batch_{datetime.now().timestamp()}",
            'total_calculations': len(calculations),
            'results': results,
            'processed_at': datetime.now().isoformat()
        }


if __name__ == "__main__":
    import sys
    
    agent = {{ agent_class_name }}(base_url=os.getenv("A2A_SERVICE_URL"))
    # Add startup logic here
'''
    
    def _generate_basic_template(self) -> str:
        """Generate basic template."""
        return '''"""
{{ agent_description }}
Generated by A2A Enhanced Agent Builder
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# A2A SDK imports
from app.a2a.sdk import (
    A2AAgentBase, a2a_handler, a2a_skill, a2a_task,
    A2AMessage, MessageRole, create_agent_id
)

{% for import_stmt in sdk_imports %}
{{ import_stmt }}
{% endfor %}

logger = logging.getLogger(__name__)


class {{ agent_class_name }}(A2AAgentBase{% for mixin in sdk_mixins %}, {{ mixin }}{% endfor %}):
    """
    {{ agent_description }}
    """
    
    def __init__(self, base_url: str, **kwargs):
        """Initialize {{ agent_name }}."""
        super().__init__(
            agent_id="{{ agent_id }}",
            name="{{ agent_name }}",
            description="{{ agent_description }}",
            version="1.0.0",
            base_url=base_url
        )
        
        # Configuration
        self.config = kwargs.get('config', {})
        
        {% if monitoring_enabled %}
        self.metrics = create_agent_metrics("{{ agent_id }}", "custom")
        {% endif %}
        
        logger.info(f"Initialized {{ agent_name }}")
    
    @a2a_skill("process")
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing method."""
        try:
            # Implement your agent logic here
            result = {
                'success': True,
                'processed_at': datetime.now().isoformat(),
                'agent_id': "{{ agent_id }}",
                'data': data
            }
            
            {% if monitoring_enabled %}
            self.metrics.record_task("process", "success")
            {% endif %}
            
            return result
        
        except Exception as e:
            {% if monitoring_enabled %}
            self.metrics.record_task("process", "failure")
            {% endif %}
            logger.error(f"Processing failed: {e}")
            raise


if __name__ == "__main__":
    import sys
    
    agent = {{ agent_class_name }}(base_url=os.getenv("A2A_SERVICE_URL"))
    # Add startup logic here
'''
    
    def _load_existing_templates(self):
        """Load existing template definitions from disk."""
        try:
            # Look for custom template files
            if self.templates_path.exists():
                for template_file in self.templates_path.glob('*.j2'):
                    if template_file.stem in ['basic_data_processor', 'advanced_validator', 'enterprise_calculator']:
                        continue  # Skip built-in templates
                    
                    # Try to load template metadata
                    metadata_file = template_file.with_suffix('.yaml')
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                template_config = yaml.safe_load(f)
                            
                            # Create AgentTemplate from config
                            template = AgentTemplate(
                                template_id=template_config['template_id'],
                                name=template_config['name'],
                                description=template_config['description'],
                                agent_type=AgentType(template_config['agent_type']),
                                template_type=TemplateType(template_config['template_type']),
                                template_path=str(template_file.name),
                                required_components=[SDKComponent(comp) for comp in template_config.get('required_components', [])],
                                optional_components=[SDKComponent(comp) for comp in template_config.get('optional_components', [])],
                                configuration_schema=template_config.get('configuration_schema', {}),
                                example_config=template_config.get('example_config', {}),
                                created_at=datetime.fromisoformat(template_config['created_at']),
                                version=template_config.get('version', '1.0.0'),
                                tags=template_config.get('tags', [])
                            )
                            
                            self.templates[template.template_id] = template
                            logger.info(f"Loaded custom template: {template.template_id}")
                            
                        except Exception as e:
                            logger.error(f"Failed to load template {template_file}: {e}")
                            
            logger.info(f"Loaded {len(self.templates)} total templates")
            
        except Exception as e:
            logger.error(f"Failed to load existing templates: {e}")
    
    async def generate_agent(self, config: AgentConfiguration) -> GenerationResult:
        """
        Generate agent from template and configuration.
        
        Args:
            config: Agent configuration
            
        Returns:
            Generation result
        """
        start_time = datetime.now()
        
        try:
            # Validate configuration
            if config.template_id not in self.templates:
                return GenerationResult(
                    success=False,
                    agent_path="",
                    generated_files=[],
                    errors=[f"Template {config.template_id} not found"]
                )
            
            template = self.templates[config.template_id]
            
            # Prepare output directory
            agent_output_path = self.output_base_path / config.agent_id
            agent_output_path.mkdir(parents=True, exist_ok=True)
            
            # Prepare template context
            context = self._prepare_template_context(config, template)
            
            # Generate main agent file
            agent_file = await self._generate_agent_file(template, context, agent_output_path)
            generated_files = [str(agent_file)]
            
            # Generate additional files
            if config.enable_tests:
                test_file = await self._generate_test_file(config, agent_output_path)
                generated_files.append(str(test_file))
            
            if config.enable_docs:
                doc_file = await self._generate_documentation(config, agent_output_path)
                generated_files.append(str(doc_file))
            
            # Generate requirements file
            requirements_file = await self._generate_requirements(config, agent_output_path)
            generated_files.append(str(requirements_file))
            
            # Generate configuration file
            config_file = await self._generate_config_file(config, agent_output_path)
            generated_files.append(str(config_file))
            
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Generated agent {config.agent_id} in {duration:.2f}s")
            
            return GenerationResult(
                success=True,
                agent_path=str(agent_output_path),
                generated_files=generated_files,
                generation_time=duration
            )
            
        except Exception as e:
            logger.error(f"Agent generation failed: {e}")
            return GenerationResult(
                success=False,
                agent_path="",
                generated_files=[],
                errors=[str(e)]
            )
    
    def _prepare_template_context(
        self,
        config: AgentConfiguration,
        template: AgentTemplate
    ) -> Dict[str, Any]:
        """Prepare context for template rendering."""
        # Collect SDK imports and mixins
        sdk_imports = []
        sdk_mixins = []
        
        for component in config.sdk_components:
            if component in self.sdk_components:
                comp_config = self.sdk_components[component]
                sdk_imports.extend(comp_config['imports'])
                sdk_mixins.extend(comp_config['mixins'])
        
        # Remove duplicates
        sdk_imports = list(set(sdk_imports))
        sdk_mixins = list(set(sdk_mixins))
        
        # Generate agent class name
        agent_class_name = ''.join(word.capitalize() for word in config.agent_id.split('_'))
        if not agent_class_name.endswith('Agent'):
            agent_class_name += 'Agent'
        
        context = {
            'agent_id': config.agent_id,
            'agent_name': config.agent_name,
            'agent_description': config.agent_description,
            'agent_class_name': agent_class_name,
            'sdk_imports': sdk_imports,
            'sdk_mixins': sdk_mixins,
            'monitoring_enabled': SDKComponent.MONITORING in config.sdk_components,
            'error_handling_enabled': SDKComponent.ERROR_HANDLING in config.sdk_components,
            'self_healing_enabled': template.template_type == TemplateType.ENTERPRISE,
            'custom_config': config.custom_config,
            'generation_timestamp': datetime.now().isoformat()
        }
        
        return context
    
    async def _generate_agent_file(
        self,
        template: AgentTemplate,
        context: Dict[str, Any],
        output_path: Path
    ) -> Path:
        """Generate main agent file."""
        template_obj = self.jinja_env.get_template(template.template_path)
        agent_code = template_obj.render(**context)
        
        agent_file = output_path / f"{context['agent_id']}.py"
        
        with open(agent_file, 'w') as f:
            f.write(agent_code)
        
        return agent_file
    
    async def _generate_test_file(self, config: AgentConfiguration, output_path: Path) -> Path:
        """Generate test file for agent."""
        test_content = f'''"""
Test suite for {config.agent_name}
Generated by A2A Enhanced Agent Builder
"""
import pytest
import asyncio
from unittest.mock import Mock, patch
from {config.agent_id} import {config.agent_id.replace('_', '').title()}Agent


class Test{config.agent_id.replace('_', '').title()}Agent:
    """Test suite for {config.agent_name}."""
    
    @pytest.fixture
    async def agent(self):
        """Create agent instance for testing."""
        return {config.agent_id.replace('_', '').title()}Agent(
            base_url=os.getenv("A2A_SERVICE_URL"),
            config={config.custom_config}
        )
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.agent_id == "{config.agent_id}"
        assert agent.name == "{config.agent_name}"
    
    @pytest.mark.asyncio
    async def test_main_functionality(self, agent):
        """Test main agent functionality."""
        # Add specific tests for your agent
        test_data = {{"test": "data"}}
        result = await agent.process(test_data)
        
        assert result["success"] is True
        assert "processed_at" in result
'''
        
        test_file = output_path / f"test_{config.agent_id}.py"
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        return test_file
    
    async def _generate_documentation(self, config: AgentConfiguration, output_path: Path) -> Path:
        """Generate documentation for agent."""
        doc_content = f'''# {config.agent_name}

{config.agent_description}

## Overview

This agent was generated using the A2A Enhanced Agent Builder.

## Configuration

Agent ID: `{config.agent_id}`
Agent Type: `{config.agent_type.value}`
Template: `{config.template_id}`

## SDK Components

The following SDK components are included:
{chr(10).join(f"- {comp.value}" for comp in config.sdk_components)}

## Usage

```python
from {config.agent_id} import {config.agent_id.replace('_', '').title()}Agent


# A2A Protocol Compliance: Require environment variables
required_env_vars = ["A2A_SERVICE_URL", "A2A_SERVICE_HOST", "A2A_BASE_URL"]
missing_vars = [var for var in required_env_vars if var in locals() and not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Required environment variables not set for A2A compliance: {missing_vars}")
# Initialize agent
agent = {config.agent_id.replace('_', '').title()}Agent(
    base_url=os.getenv("A2A_SERVICE_URL"),
    config={config.custom_config}
)

# Use agent functionality
result = await agent.process(data)
```

## Configuration Options

```json
{json.dumps(config.custom_config, indent=2)}
```

## Generated Files

- `{config.agent_id}.py` - Main agent implementation
- `test_{config.agent_id}.py` - Test suite
- `requirements.txt` - Dependencies
- `config.yaml` - Configuration file
- `README.md` - This documentation

## Support

For support and documentation, visit the A2A project repository.
'''
        
        doc_file = output_path / "README.md"
        
        with open(doc_file, 'w') as f:
            f.write(doc_content)
        
        return doc_file
    
    async def _generate_requirements(self, config: AgentConfiguration, output_path: Path) -> Path:
        """Generate requirements.txt file."""
        base_requirements = [
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
            "pydantic>=1.8.0",
            "httpx>=0.24.0",
            "asyncio-mqtt>=0.10.0"
        ]
        
        # Add SDK component dependencies
        sdk_requirements = set()
        for component in config.sdk_components:
            if component in self.sdk_components:
                deps = self.sdk_components[component]['dependencies']
                sdk_requirements.update(deps)
        
        all_requirements = base_requirements + sorted(list(sdk_requirements))
        
        requirements_file = output_path / "requirements.txt"
        
        with open(requirements_file, 'w') as f:
            f.write('\n'.join(all_requirements))
        
        return requirements_file
    
    async def _generate_config_file(self, config: AgentConfiguration, output_path: Path) -> Path:
        """Generate configuration file."""
        config_data = {
            'agent': {
                'id': config.agent_id,
                'name': config.agent_name,
                'description': config.agent_description,
                'type': config.agent_type.value,
                'version': '1.0.0'
            },
            'sdk_components': [comp.value for comp in config.sdk_components],
            'configuration': config.custom_config,
            'generated': {
                'timestamp': datetime.now().isoformat(),
                'template': config.template_id,
                'builder_version': '1.0.0'
            }
        }
        
        config_file = output_path / "config.yaml"
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
        
        return config_file
    
    def list_templates(self) -> List[AgentTemplate]:
        """List available templates."""
        return list(self.templates.values())
    
    def get_template(self, template_id: str) -> Optional[AgentTemplate]:
        """Get template by ID."""
        return self.templates.get(template_id)
    
    def validate_configuration(self, config: AgentConfiguration) -> List[str]:
        """Validate agent configuration."""
        errors = []
        
        # Check template exists
        if config.template_id not in self.templates:
            errors.append(f"Template {config.template_id} not found")
            return errors
        
        template = self.templates[config.template_id]
        
        # Check required components
        missing_required = set(template.required_components) - set(config.sdk_components)
        if missing_required:
            errors.append(f"Missing required SDK components: {missing_required}")
        
        # Validate agent ID format
        if not config.agent_id.replace('_', '').isalnum():
            errors.append("Agent ID must contain only alphanumeric characters and underscores")
        
        # Check output path
        if not config.output_path:
            errors.append("Output path is required")
        
        return errors