"""
MCP Validation Tools
Common validation, schema compliance, and integrity checking tools
"""

import jsonschema
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import re
from datetime import datetime
import hashlib
from collections import defaultdict
from ..sdk.mcpDecorators import mcp_tool, mcp_resource, mcp_prompt
from ..sdk.mcpSkillCoordination import skill_provides, skill_depends_on

logger = logging.getLogger(__name__)


class MCPValidationTools:
    """MCP-enabled validation tools for schema compliance and data integrity"""
    
    def __init__(self):
        self.validation_levels = ["strict", "standard", "lenient"]
        
        # Common validation patterns
        self.validation_patterns = {
            "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            "phone": r'^\+?[\d\s\-\(\)]{10,}$',
            "url": r'^https?://[^\s/$.?#].[^\s]*$',
            "date_iso": r'^\d{4}-\d{2}-\d{2}$',
            "datetime_iso": r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
            "uuid": r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            "ipv4": r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$',
            "alphanumeric": r'^[a-zA-Z0-9]+$',
            "numeric": r'^[0-9]+$'
        }
        
        # Common business rule types
        self.rule_types = [
            "required_field",
            "format_validation",
            "range_check",
            "dependency_check",
            "uniqueness_check",
            "referential_integrity",
            "business_logic"
        ]
    
    @mcp_tool(
        name="validate_schema_compliance",
        description="Comprehensive schema validation with multiple validation levels",
        input_schema={
            "type": "object",
            "properties": {
                "data": {
                    "description": "Data to validate against schema"
                },
                "schema": {
                    "type": "object",
                    "description": "JSON schema for validation"
                },
                "validation_level": {
                    "type": "string",
                    "enum": ["strict", "standard", "lenient"],
                    "default": "standard",
                    "description": "Validation strictness level"
                },
                "custom_validators": {
                    "type": "object",
                    "description": "Custom validation functions",
                    "additionalProperties": {"type": "string"}
                },
                "return_details": {
                    "type": "boolean",
                    "default": True,
                    "description": "Return detailed validation results"
                }
            },
            "required": ["data", "schema"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "is_valid": {"type": "boolean"},
                "compliance_score": {"type": "number"},
                "errors": {"type": "array"},
                "warnings": {"type": "array"},
                "field_validations": {"type": "object"}
            }
        }
    )
    @skill_provides("schema_validation", "compliance_checking")
    async def validate_schema_compliance(self,
                                   data: Any,
                                   schema: Dict[str, Any],
                                   validation_level: str = "standard",
                                   custom_validators: Optional[Dict[str, str]] = None,
                                   return_details: bool = True) -> Dict[str, Any]:
        """Validate data against JSON schema with configurable strictness"""
        
        errors = []
        warnings = []
        field_validations = {}
        
        try:
            # Primary JSON schema validation
            jsonschema.validate(instance=data, schema=schema)
            primary_valid = True
        except jsonschema.ValidationError as e:
            primary_valid = False
            if validation_level == "strict":
                errors.append(f"Schema validation failed: {e.message}")
            elif validation_level == "standard":
                if e.validator in ["required", "type"]:
                    errors.append(f"Critical validation error: {e.message}")
                else:
                    warnings.append(f"Schema warning: {e.message}")
            else:  # lenient
                warnings.append(f"Schema suggestion: {e.message}")
        except Exception as e:
            errors.append(f"Schema validation error: {str(e)}")
            primary_valid = False
        
        # Additional custom validations
        if custom_validators:
            custom_results = await self._apply_custom_validators(
                data, custom_validators, validation_level
            )
            errors.extend(custom_results["errors"])
            warnings.extend(custom_results["warnings"])
            field_validations.update(custom_results["field_validations"])
        
        # Field-level validation details
        if return_details and isinstance(data, dict):
            field_validations.update(
                await self._validate_fields_detailed(data, schema, validation_level)
            )
        
        # Calculate compliance score
        total_checks = 1 + len(custom_validators or {}) + len(field_validations)
        passed_checks = (1 if primary_valid else 0) + sum(
            1 for fv in field_validations.values() if fv.get("valid", False)
        )
        compliance_score = passed_checks / total_checks if total_checks > 0 else 0.0
        
        # Determine overall validity
        is_valid = compliance_score >= self._get_validity_threshold(validation_level)
        
        result = {
            "is_valid": is_valid,
            "compliance_score": compliance_score,
            "errors": errors,
            "validation_errors": errors,  # Alias for compatibility
            "warnings": warnings,
            "validation_level": validation_level,
            "total_checks": total_checks,
            "passed_checks": passed_checks
        }
        
        if return_details:
            result["field_validations"] = field_validations
        
        return result
    
    @mcp_tool(
        name="validate_data_structure",
        description="Validate data structure and format compliance",
        input_schema={
            "type": "object",
            "properties": {
                "data": {
                    "description": "Data to validate structure"
                },
                "expected_structure": {
                    "type": "object",
                    "description": "Expected data structure definition"
                },
                "validation_level": {
                    "type": "string",
                    "enum": ["strict", "standard", "lenient"],
                    "default": "standard"
                }
            },
            "required": ["data"]
        }
    )
    async def validate_data_structure(self,
                                    data: Any,
                                    expected_structure: Optional[Dict[str, Any]] = None,
                                    validation_level: str = "standard") -> Dict[str, Any]:
        """Validate data structure and format compliance"""
        
        errors = []
        warnings = []
        is_valid = True
        
        try:
            # Basic structure validation
            if expected_structure:
                if isinstance(data, dict) and isinstance(expected_structure, dict):
                    # Check required fields
                    required_fields = expected_structure.get("required", [])
                    for field in required_fields:
                        if field not in data:
                            errors.append(f"Missing required field: {field}")
                            is_valid = False
                    
                    # Check field types
                    for field, value in data.items():
                        if field in expected_structure.get("properties", {}):
                            expected_type = expected_structure["properties"][field].get("type")
                            if expected_type and not self._check_type_compatibility(value, expected_type):
                                errors.append(f"Field {field} type mismatch: expected {expected_type}, got {type(value).__name__}")
                                if validation_level == "strict":
                                    is_valid = False
                                elif validation_level == "standard":
                                    warnings.append(f"Type conversion may be needed for field {field}")
            
            # Basic data integrity checks
            if isinstance(data, dict):
                for key, value in data.items():
                    if value is None and validation_level == "strict":
                        warnings.append(f"Null value in field {key}")
                    
                    # Check for empty strings
                    if isinstance(value, str) and len(value.strip()) == 0:
                        warnings.append(f"Empty string in field {key}")
            
            # Update validity based on validation level
            if validation_level == "lenient":
                is_valid = len(errors) == 0
            elif validation_level == "standard":
                is_valid = len(errors) == 0
            else:  # strict
                is_valid = len(errors) == 0 and len(warnings) == 0
            
        except Exception as e:
            errors.append(f"Structure validation failed: {str(e)}")
            is_valid = False
        
        return {
            "is_valid": is_valid,
            "validation_errors": errors,
            "warnings": warnings,
            "validation_level": validation_level,
            "structure_checks_passed": len(errors) == 0,
            "total_issues": len(errors) + len(warnings)
        }
    
    @mcp_tool(
        name="verify_data_integrity",
        description="Multi-layered data integrity verification",
        input_schema={
            "type": "object",
            "properties": {
                "data": {
                    "description": "Data to verify integrity"
                },
                "integrity_checks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["checksum", "format", "relationships"],
                    "description": "Types of integrity checks to perform"
                },
                "baseline": {
                    "description": "Baseline data for comparison"
                },
                "tolerance": {
                    "type": "number",
                    "default": 0.05,
                    "description": "Tolerance for numerical comparisons"
                }
            },
            "required": ["data"]
        }
    )
    @skill_provides("data_integrity", "consistency_verification")
    async def verify_data_integrity(self,
                              data: Any,
                              integrity_checks: List[str] = None,
                              baseline: Optional[Any] = None,
                              tolerance: float = 0.05) -> Dict[str, Any]:
        """Comprehensive data integrity verification"""
        
        integrity_checks = integrity_checks or ["checksum", "format", "relationships"]
        check_results = {}
        anomalies = []
        
        for check_type in integrity_checks:
            if check_type == "checksum":
                result = await self._verify_checksum_integrity(data, baseline)
            elif check_type == "format":
                result = await self._verify_format_integrity(data)
            elif check_type == "relationships":
                result = await self._verify_relationship_integrity(data)
            elif check_type == "numerical":
                result = await self._verify_numerical_integrity(data, baseline, tolerance)
            elif check_type == "temporal":
                result = await self._verify_temporal_integrity(data)
            else:
                result = {"valid": False, "message": f"Unknown check type: {check_type}"}
            
            check_results[check_type] = result
            
            if not result.get("valid", False):
                anomalies.extend(result.get("anomalies", []))
        
        # Calculate overall integrity score
        valid_checks = sum(1 for result in check_results.values() if result.get("valid", False))
        integrity_score = valid_checks / len(integrity_checks) if integrity_checks else 0.0
        
        return {
            "integrity_score": integrity_score,
            "check_results": check_results,
            "anomalies": anomalies,
            "checks_performed": integrity_checks,
            "is_integrity_valid": integrity_score >= 0.8
        }
    
    @mcp_tool(
        name="validate_business_rules",
        description="Business rule validation engine with configurable rules",
        input_schema={
            "type": "object",
            "properties": {
                "data": {
                    "description": "Data to validate against business rules"
                },
                "rules": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "rule_id": {"type": "string"},
                            "rule_type": {"type": "string"},
                            "condition": {"type": "string"},
                            "action": {"type": "string"},
                            "severity": {"type": "string"}
                        }
                    },
                    "description": "Business rules to validate"
                },
                "context": {
                    "type": "object",
                    "description": "Additional context for rule evaluation"
                },
                "enforcement_level": {
                    "type": "string",
                    "enum": ["error", "warning", "info"],
                    "default": "error",
                    "description": "How strictly to enforce rule violations"
                }
            },
            "required": ["data", "rules"]
        }
    )
    @skill_provides("business_validation", "rule_enforcement")
    async def validate_business_rules(self,
                                data: Any,
                                rules: List[Dict[str, Any]],
                                context: Optional[Dict[str, Any]] = None,
                                enforcement_level: str = "error") -> Dict[str, Any]:
        """Validate data against configurable business rules"""
        
        context = context or {}
        violations = []
        rule_results = {}
        compliance_score = 0.0
        
        for rule in rules:
            rule_id = rule.get("rule_id", f"rule_{len(rule_results)}")
            rule_type = rule.get("rule_type", "custom")
            condition = rule.get("condition", "")
            severity = rule.get("severity", enforcement_level)
            
            # Evaluate the rule
            try:
                if rule_type == "required_field":
                    result = await self._validate_required_field_rule(data, rule, context)
                elif rule_type == "format_validation":
                    result = await self._validate_format_rule(data, rule, context)
                elif rule_type == "range_check":
                    result = await self._validate_range_rule(data, rule, context)
                elif rule_type == "dependency_check":
                    result = await self._validate_dependency_rule(data, rule, context)
                elif rule_type == "uniqueness_check":
                    result = await self._validate_uniqueness_rule(data, rule, context)
                elif rule_type == "business_logic":
                    result = await self._validate_business_logic_rule(data, rule, context)
                else:
                    result = {"valid": False, "message": f"Unknown rule type: {rule_type}"}
                
                rule_results[rule_id] = result
                
                if result.get("valid", False):
                    compliance_score += 1
                else:
                    violations.append({
                        "rule_id": rule_id,
                        "rule_type": rule_type,
                        "severity": severity,
                        "message": result.get("message", "Rule violation"),
                        "details": result.get("details", {})
                    })
                    
            except Exception as e:
                rule_results[rule_id] = {
                    "valid": False,
                    "message": f"Rule evaluation error: {str(e)}"
                }
                violations.append({
                    "rule_id": rule_id,
                    "severity": "error",
                    "message": f"Rule evaluation failed: {str(e)}"
                })
        
        # Calculate final compliance score
        compliance_score = compliance_score / len(rules) if rules else 1.0
        
        return {
            "compliance_score": compliance_score,
            "violations": violations,
            "rule_results": rule_results,
            "total_rules": len(rules),
            "passed_rules": len(rules) - len(violations),
            "enforcement_level": enforcement_level
        }
    
    @mcp_resource(
        uri="validation://patterns/common",
        name="Common Validation Patterns",
        description="Library of common validation patterns and formats",
        mime_type="application/json"
    )
    async def get_validation_patterns(self) -> Dict[str, Any]:
        """Get common validation patterns and their descriptions"""
        return {
            "patterns": {
                pattern_name: {
                    "regex": pattern,
                    "description": self._get_pattern_description(pattern_name),
                    "examples": self._get_pattern_examples(pattern_name)
                }
                for pattern_name, pattern in self.validation_patterns.items()
            },
            "rule_types": {
                rule_type: self._get_rule_type_description(rule_type)
                for rule_type in self.rule_types
            },
            "validation_levels": {
                "strict": "All validations must pass, no tolerance for errors",
                "standard": "Critical validations must pass, warnings allowed",
                "lenient": "Validation suggestions provided, minimal enforcement"
            }
        }
    
    @mcp_prompt(
        name="validation_report",
        description="Generate comprehensive validation report",
        arguments=[
            {
                "name": "validation_results",
                "description": "Results from validation operations",
                "required": True
            },
            {
                "name": "report_style",
                "description": "Style of report (technical, business, executive)",
                "required": False
            }
        ]
    )
    async def validation_report(self,
                          validation_results: Dict[str, Any],
                          report_style: str = "technical") -> str:
        """Generate comprehensive validation report"""
        
        report = f"# Validation Report\n\n"
        
        # Schema validation results
        if "compliance_score" in validation_results:
            score = validation_results["compliance_score"]
            report += f"**Compliance Score**: {score:.2%}\n"
            report += f"**Validation Status**: {'✅ PASSED' if validation_results.get('is_valid', False) else '❌ FAILED'}\n\n"
        
        # Business rule validation results
        if "violations" in validation_results:
            violations = validation_results["violations"]
            if violations:
                report += f"## Rule Violations ({len(violations)})\n\n"
                for violation in violations:
                    severity_icon = "🔴" if violation["severity"] == "error" else "🟡" if violation["severity"] == "warning" else "🔵"
                    report += f"{severity_icon} **{violation['rule_id']}**: {violation['message']}\n"
            else:
                report += "## ✅ No Rule Violations\n\n"
        
        # Integrity check results
        if "check_results" in validation_results:
            report += "## Integrity Checks\n\n"
            for check_type, result in validation_results["check_results"].items():
                status = "✅" if result.get("valid", False) else "❌"
                report += f"- **{check_type.title()}**: {status} {result.get('message', '')}\n"
        
        # Recommendations based on report style
        if report_style == "business":
            report += self._generate_business_recommendations(validation_results)
        elif report_style == "executive":
            report += self._generate_executive_summary(validation_results)
        else:  # technical
            report += self._generate_technical_details(validation_results)
        
        return report
    
    # Internal helper methods
    async def _apply_custom_validators(self, data: Any, validators: Dict[str, str], 
                                 level: str) -> Dict[str, Any]:
        """Apply custom validation functions"""
        errors = []
        warnings = []
        field_validations = {}
        
        # Real custom validator implementation
        for validator_name, validator_code in validators.items():
            try:
                # Execute custom validation logic
                if validator_name == "email_validator":
                    for field, value in data.items():
                        if isinstance(value, str) and "@" in value:
                            if not re.match(self.validation_patterns["email"], value):
                                errors.append(f"Invalid email format in field {field}: {value}")
                                field_validations[field] = {"valid": False, "error": "Invalid email format"}
                            else:
                                field_validations[field] = {"valid": True}
                
                elif validator_name == "phone_validator":
                    for field, value in data.items():
                        if isinstance(value, str) and any(char.isdigit() for char in value):
                            if not re.match(self.validation_patterns["phone"], value):
                                errors.append(f"Invalid phone format in field {field}: {value}")
                                field_validations[field] = {"valid": False, "error": "Invalid phone format"}
                            else:
                                field_validations[field] = {"valid": True}
                
                elif validator_name == "date_validator":
                    for field, value in data.items():
                        if isinstance(value, str) and "-" in value:
                            if not re.match(self.validation_patterns["date_iso"], value):
                                errors.append(f"Invalid date format in field {field}: {value}")
                                field_validations[field] = {"valid": False, "error": "Invalid date format"}
                            else:
                                field_validations[field] = {"valid": True}
                
                elif validator_name == "range_validator":
                    # Parse range from validator_code (e.g., "min:0,max:100")
                    if "min:" in validator_code and "max:" in validator_code:
                        parts = validator_code.split(",")
                        min_val = float(parts[0].split(":")[1])
                        max_val = float(parts[1].split(":")[1])
                        
                        for field, value in data.items():
                            if isinstance(value, (int, float)):
                                if not (min_val <= value <= max_val):
                                    errors.append(f"Value {value} in field {field} outside range [{min_val}, {max_val}]")
                                    field_validations[field] = {"valid": False, "error": f"Outside range [{min_val}, {max_val}]"}
                                else:
                                    field_validations[field] = {"valid": True}
                
                elif validator_name == "length_validator":
                    # Parse length from validator_code (e.g., "min:2,max:50")
                    if "min:" in validator_code and "max:" in validator_code:
                        parts = validator_code.split(",")
                        min_len = int(parts[0].split(":")[1])
                        max_len = int(parts[1].split(":")[1])
                        
                        for field, value in data.items():
                            if isinstance(value, str):
                                if not (min_len <= len(value) <= max_len):
                                    errors.append(f"String length {len(value)} in field {field} outside range [{min_len}, {max_len}]")
                                    field_validations[field] = {"valid": False, "error": f"Length outside range [{min_len}, {max_len}]"}
                                else:
                                    field_validations[field] = {"valid": True}
                
                else:
                    # SECURITY FIX: Use safe validation instead of eval()
                    import ast
                    import operator
                    
                    # Define safe operations for validation expressions
                    SAFE_OPERATORS = {
                        ast.Add: operator.add,
                        ast.Sub: operator.sub,
                        ast.Mult: operator.mul,
                        ast.Div: operator.truediv,
                        ast.Mod: operator.mod,
                        ast.Eq: operator.eq,
                        ast.NotEq: operator.ne,
                        ast.Lt: operator.lt,
                        ast.LtE: operator.le,
                        ast.Gt: operator.gt,
                        ast.GtE: operator.ge,
                        ast.And: operator.and_,
                        ast.Or: operator.or_,
                        ast.Not: operator.not_,
                        ast.In: lambda a, b: a in b,
                        ast.NotIn: lambda a, b: a not in b,
                    }
                    
                    def safe_eval_expression(node, data_context):
                        """Safely evaluate validation expressions without eval()"""
                        if isinstance(node, ast.Constant):
                            return node.value
                        elif isinstance(node, ast.Name):
                            if node.id == 'data':
                                return data_context
                            elif node.id in {'True', 'False', 'None'}:
                                return {'True': True, 'False': False, 'None': None}[node.id]
                            else:
                                raise ValueError(f"Unsafe variable access: {node.id}")
                        elif isinstance(node, ast.Compare):
                            left = safe_eval_expression(node.left, data_context)
                            for op, comparator in zip(node.ops, node.comparators):
                                right = safe_eval_expression(comparator, data_context)
                                if type(op) not in SAFE_OPERATORS:
                                    raise ValueError(f"Unsafe operation: {type(op)}")
                                left = SAFE_OPERATORS[type(op)](left, right)
                            return left
                        elif isinstance(node, ast.BinOp):
                            left = safe_eval_expression(node.left, data_context)
                            right = safe_eval_expression(node.right, data_context)
                            if type(node.op) not in SAFE_OPERATORS:
                                raise ValueError(f"Unsafe operation: {type(node.op)}")
                            return SAFE_OPERATORS[type(node.op)](left, right)
                        else:
                            raise ValueError(f"Unsafe AST node: {type(node)}")
                    
                    try:
                        # Parse and validate expression safety
                        parsed = ast.parse(validator_code, mode='eval')
                        
                        # Check for dangerous nodes
                        dangerous_nodes = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, 
                                         ast.Call, ast.Attribute, ast.Subscript)
                        if any(isinstance(node, dangerous_nodes) for node in ast.walk(parsed)):
                            warnings.append(f"Custom validator {validator_name} contains unsafe operations")
                        else:
                            # Safe evaluation
                            result = safe_eval_expression(parsed.body, data)
                            if not result:
                                errors.append(f"Custom validator {validator_name} failed")
                                
                    except Exception as validation_error:
                        warnings.append(f"Custom validator {validator_name} failed: {str(validation_error)}")
                        
            except Exception as e:
                errors.append(f"Error executing custom validator {validator_name}: {str(e)}")
        
        return {
            "errors": errors,
            "warnings": warnings,
            "field_validations": field_validations
        }
    
    async def _validate_fields_detailed(self, data: Dict[str, Any], 
                                  schema: Dict[str, Any], level: str) -> Dict[str, Any]:
        """Perform detailed field-level validation"""
        field_validations = {}
        
        properties = schema.get("properties", {})
        for field, field_schema in properties.items():
            if field in data:
                validation_result = await self._validate_single_field(
                    field, data[field], field_schema, level
                )
                field_validations[field] = validation_result
        
        return field_validations
    
    async def _validate_single_field(self, field_name: str, value: Any, 
                                field_schema: Dict[str, Any], level: str) -> Dict[str, Any]:
        """Validate a single field against its schema"""
        result = {"valid": True, "messages": []}
        
        # Type validation
        expected_type = field_schema.get("type")
        if expected_type:
            if not self._check_type_compatibility(value, expected_type):
                result["valid"] = False
                result["messages"].append(f"Expected type {expected_type}, got {type(value).__name__}")
        
        # Format validation
        format_type = field_schema.get("format")
        if format_type and isinstance(value, str):
            if format_type in self.validation_patterns:
                pattern = self.validation_patterns[format_type]
                if not re.match(pattern, value):
                    if level == "strict":
                        result["valid"] = False
                    result["messages"].append(f"Format validation failed for {format_type}")
        
        return result
    
    def _check_type_compatibility(self, value: Any, expected_type: str) -> bool:
        """Check if value type is compatible with expected type"""
        type_mapping = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True  # Unknown type, assume valid
    
    def _get_validity_threshold(self, validation_level: str) -> float:
        """Get validity threshold based on validation level"""
        thresholds = {
            "strict": 1.0,
            "standard": 0.8,
            "lenient": 0.6
        }
        return thresholds.get(validation_level, 0.8)
    
    # Integrity verification methods
    async def _verify_checksum_integrity(self, data: Any, baseline: Optional[Any]) -> Dict[str, Any]:
        """Verify data integrity using checksums"""
        try:
            data_str = str(data)
            current_hash = hashlib.sha256(data_str.encode()).hexdigest()
            
            if baseline is not None:
                baseline_str = str(baseline)
                baseline_hash = hashlib.sha256(baseline_str.encode()).hexdigest()
                valid = current_hash == baseline_hash
                message = "Checksum matches baseline" if valid else "Checksum mismatch detected"
            else:
                valid = True
                message = f"Checksum calculated: {current_hash[:8]}..."
            
            return {
                "valid": valid,
                "message": message,
                "checksum": current_hash,
                "anomalies": [] if valid else ["checksum_mismatch"]
            }
        except Exception as e:
            return {
                "valid": False,
                "message": f"Checksum calculation failed: {str(e)}",
                "anomalies": ["checksum_calculation_error"]
            }
    
    async def _verify_format_integrity(self, data: Any) -> Dict[str, Any]:
        """Verify format integrity of data"""
        anomalies = []
        
        if isinstance(data, dict):
            # Check for malformed keys or values
            for key, value in data.items():
                if not isinstance(key, str):
                    anomalies.append(f"Non-string key: {key}")
                if value is None:
                    anomalies.append(f"Null value for key: {key}")
        
        return {
            "valid": len(anomalies) == 0,
            "message": f"Format check {'passed' if len(anomalies) == 0 else 'failed'}",
            "anomalies": anomalies
        }
    
    async def _verify_relationship_integrity(self, data: Any) -> Dict[str, Any]:
        """Verify referential integrity and relationships"""
        anomalies = []
        
        if isinstance(data, dict):
            # Check for orphaned references (foreign keys without corresponding parent)
            for key, value in data.items():
                if key.endswith("_id") and value is not None:
                    # Check if referenced ID looks valid
                    if isinstance(value, str) and len(value) == 0:
                        anomalies.append(f"Empty reference ID in field: {key}")
                    elif isinstance(value, (int, float)) and value <= 0:
                        anomalies.append(f"Invalid reference ID in field: {key} (value: {value})")
                
                # Check for circular references in nested objects
                if isinstance(value, dict) and key in value:
                    anomalies.append(f"Potential circular reference detected: {key}")
                
                # Check for broken email relationships
                if key == "email" and isinstance(value, str):
                    if "@" in value and not re.match(self.validation_patterns["email"], value):
                        anomalies.append(f"Malformed email relationship: {value}")
        
        elif isinstance(data, list):
            # Check for duplicate IDs in list
            ids_seen = set()
            for i, item in enumerate(data):
                if isinstance(item, dict) and "id" in item:
                    item_id = item["id"]
                    if item_id in ids_seen:
                        anomalies.append(f"Duplicate ID found at index {i}: {item_id}")
                    ids_seen.add(item_id)
        
        return {
            "valid": len(anomalies) == 0,
            "message": f"Relationship integrity {'verified' if len(anomalies) == 0 else 'violations found'}",
            "anomalies": anomalies
        }
    
    async def _verify_numerical_integrity(self, data: Any, baseline: Optional[Any], 
                                    tolerance: float) -> Dict[str, Any]:
        """Verify numerical data integrity"""
        anomalies = []
        
        if isinstance(data, (int, float)) and isinstance(baseline, (int, float)):
            difference = abs(data - baseline)
            relative_diff = difference / abs(baseline) if baseline != 0 else difference
            
            if relative_diff > tolerance:
                anomalies.append(f"Numerical deviation exceeds tolerance: {relative_diff:.4f} > {tolerance}")
        
        return {
            "valid": len(anomalies) == 0,
            "message": f"Numerical integrity {'verified' if len(anomalies) == 0 else 'violated'}",
            "anomalies": anomalies
        }
    
    async def _verify_temporal_integrity(self, data: Any) -> Dict[str, Any]:
        """Verify temporal data integrity"""
        anomalies = []
        
        if isinstance(data, dict):
            date_fields = []
            # Find date/time fields
            for key, value in data.items():
                if isinstance(value, str):
                    # Check for date formats
                    if re.match(self.validation_patterns["date_iso"], value):
                        date_fields.append((key, value, "date"))
                    elif re.match(self.validation_patterns["datetime_iso"], value):
                        date_fields.append((key, value, "datetime"))
                    elif key.lower() in ["created_at", "updated_at", "timestamp", "date", "time"]:
                        date_fields.append((key, value, "timestamp"))
            
            # Validate temporal relationships
            from datetime import datetime
            current_time = datetime.now()
            
            for field, value, field_type in date_fields:
                try:
                    if field_type == "date":
                        parsed_date = datetime.fromisoformat(value)
                    elif field_type == "datetime":
                        parsed_date = datetime.fromisoformat(value.replace('Z', '+00:00'))
                    else:
                        # Try parsing various timestamp formats
                        for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
                            try:
                                parsed_date = datetime.strptime(value, fmt)
                                break
                            except:
                                continue
                        else:
                            anomalies.append(f"Unparseable timestamp in field {field}: {value}")
                            continue
                    
                    # Check for future dates in past contexts
                    if field.lower() in ["created_at", "born", "established"] and parsed_date > current_time:
                        anomalies.append(f"Future date in past context field {field}: {value}")
                    
                    # Check for very old dates that might be errors
                    import datetime as dt
                    if parsed_date < datetime(1900, 1, 1):
                        anomalies.append(f"Suspiciously old date in field {field}: {value}")
                    
                    # Check for updated_at before created_at
                    if field == "updated_at" and "created_at" in data:
                        try:
                            created = datetime.fromisoformat(str(data["created_at"]).replace('Z', '+00:00'))
                            if parsed_date < created:
                                anomalies.append(f"Updated time {value} before created time {data['created_at']}")
                        except:
                            pass
                            
                except Exception as e:
                    anomalies.append(f"Error parsing temporal field {field}: {str(e)}")
        
        elif isinstance(data, list):
            # Check temporal ordering in time series data
            timestamps = []
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    for key in ["timestamp", "created_at", "date", "time"]:
                        if key in item:
                            timestamps.append((i, item[key]))
                            break
            
            # Verify chronological order
            for i in range(1, len(timestamps)):
                prev_idx, prev_time = timestamps[i-1]
                curr_idx, curr_time = timestamps[i]
                
                try:
                    from datetime import datetime
                    prev_dt = datetime.fromisoformat(str(prev_time).replace('Z', '+00:00'))
                    curr_dt = datetime.fromisoformat(str(curr_time).replace('Z', '+00:00'))
                    
                    if curr_dt < prev_dt:
                        anomalies.append(f"Temporal order violation: item {curr_idx} ({curr_time}) before item {prev_idx} ({prev_time})")
                except:
                    pass
        
        return {
            "valid": len(anomalies) == 0,
            "message": f"Temporal integrity {'verified' if len(anomalies) == 0 else 'violations found'}",
            "anomalies": anomalies
        }
    
    # Business rule validation methods
    async def _validate_required_field_rule(self, data: Any, rule: Dict[str, Any], 
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate required field business rule"""
        field_name = rule.get("field", "")
        if isinstance(data, dict) and field_name in data and data[field_name] is not None:
            return {"valid": True, "message": f"Required field '{field_name}' is present"}
        else:
            return {"valid": False, "message": f"Required field '{field_name}' is missing or null"}
    
    async def _validate_format_rule(self, data: Any, rule: Dict[str, Any], 
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate format business rule"""
        field_name = rule.get("field")
        pattern = rule.get("pattern")
        format_type = rule.get("format_type")
        
        if not field_name:
            return {"valid": False, "message": "Format rule missing field specification"}
        
        if isinstance(data, dict) and field_name in data:
            value = data[field_name]
            
            if value is None:
                return {"valid": True, "message": f"Field '{field_name}' is null (format validation skipped)"}
            
            value_str = str(value)
            
            # Use predefined patterns if format_type specified
            if format_type and format_type in self.validation_patterns:
                pattern = self.validation_patterns[format_type]
            
            if pattern:
                if re.match(pattern, value_str):
                    return {"valid": True, "message": f"Field '{field_name}' matches required format"}
                else:
                    return {"valid": False, "message": f"Field '{field_name}' does not match format: {pattern}"}
            else:
                return {"valid": False, "message": "Format rule missing pattern specification"}
        else:
            return {"valid": False, "message": f"Field '{field_name}' not found in data"}
    
    async def _validate_range_rule(self, data: Any, rule: Dict[str, Any], 
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate range business rule"""
        field_name = rule.get("field")
        min_value = rule.get("min")
        max_value = rule.get("max")
        
        if not field_name:
            return {"valid": False, "message": "Range rule missing field specification"}
        
        if isinstance(data, dict) and field_name in data:
            value = data[field_name]
            
            if value is None:
                return {"valid": True, "message": f"Field '{field_name}' is null (range validation skipped)"}
            
            try:
                # Try to convert to numeric for comparison
                if isinstance(value, str):
                    # Check if it's a numeric string
                    if '.' in value:
                        numeric_value = float(value)
                    else:
                        numeric_value = int(value)
                else:
                    numeric_value = float(value)
                
                # Check minimum value
                if min_value is not None and numeric_value < min_value:
                    return {"valid": False, "message": f"Field '{field_name}' value {numeric_value} is below minimum {min_value}"}
                
                # Check maximum value
                if max_value is not None and numeric_value > max_value:
                    return {"valid": False, "message": f"Field '{field_name}' value {numeric_value} exceeds maximum {max_value}"}
                
                return {"valid": True, "message": f"Field '{field_name}' value {numeric_value} is within valid range"}
                
            except (ValueError, TypeError):
                return {"valid": False, "message": f"Field '{field_name}' value cannot be converted to numeric for range validation"}
        else:
            return {"valid": False, "message": f"Field '{field_name}' not found in data"}
    
    async def _validate_dependency_rule(self, data: Any, rule: Dict[str, Any], 
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dependency business rule"""
        field_name = rule.get("field")
        depends_on = rule.get("depends_on")
        condition = rule.get("condition", "required_if_present")
        
        if not field_name or not depends_on:
            return {"valid": False, "message": "Dependency rule missing field or depends_on specification"}
        
        if not isinstance(data, dict):
            return {"valid": False, "message": "Dependency validation requires dictionary data"}
        
        dependency_value = data.get(depends_on)
        field_value = data.get(field_name)
        
        if condition == "required_if_present":
            # Field is required if dependency field is present and not null
            if dependency_value is not None and field_value is None:
                return {"valid": False, "message": f"Field '{field_name}' is required when '{depends_on}' is present"}
        
        elif condition == "required_if_value":
            # Field is required if dependency field has specific value
            required_value = rule.get("required_value")
            if dependency_value == required_value and field_value is None:
                return {"valid": False, "message": f"Field '{field_name}' is required when '{depends_on}' equals '{required_value}'"}
        
        elif condition == "mutually_exclusive":
            # Fields cannot both be present
            if dependency_value is not None and field_value is not None:
                return {"valid": False, "message": f"Fields '{field_name}' and '{depends_on}' are mutually exclusive"}
        
        elif condition == "both_or_neither":
            # Both fields must be present or both must be absent
            if (dependency_value is None) != (field_value is None):
                return {"valid": False, "message": f"Fields '{field_name}' and '{depends_on}' must both be present or both be absent"}
        
        return {"valid": True, "message": f"Dependency rule satisfied for field '{field_name}'"}
    
    async def _validate_uniqueness_rule(self, data: Any, rule: Dict[str, Any], 
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate uniqueness business rule"""
        field_name = rule.get("field")
        scope = rule.get("scope", "global")  # global, dataset, or group
        
        if not field_name:
            return {"valid": False, "message": "Uniqueness rule missing field specification"}
        
        if isinstance(data, dict) and field_name in data:
            value = data[field_name]
            
            if value is None:
                return {"valid": True, "message": f"Field '{field_name}' is null (uniqueness validation skipped)"}
            
            # Get comparison dataset from context
            comparison_data = context.get("comparison_dataset", [])
            
            if scope == "dataset" and comparison_data:
                # Check uniqueness within provided dataset
                value_count = 0
                for item in comparison_data:
                    if isinstance(item, dict) and item.get(field_name) == value:
                        value_count += 1
                
                if value_count > 1:
                    return {"valid": False, "message": f"Field '{field_name}' value '{value}' is not unique in dataset (found {value_count} occurrences)"}
                else:
                    return {"valid": True, "message": f"Field '{field_name}' value '{value}' is unique in dataset"}
            
            elif scope == "group":
                # Check uniqueness within a group field
                group_field = rule.get("group_field")
                if group_field and comparison_data:
                    current_group = data.get(group_field)
                    if current_group is not None:
                        value_count = 0
                        for item in comparison_data:
                            if (isinstance(item, dict) and 
                                item.get(group_field) == current_group and 
                                item.get(field_name) == value):
                                value_count += 1
                        
                        if value_count > 1:
                            return {"valid": False, "message": f"Field '{field_name}' value '{value}' is not unique within group '{group_field}={current_group}'"}
                        else:
                            return {"valid": True, "message": f"Field '{field_name}' value '{value}' is unique within group"}
            
            # Default: assume uniqueness constraint is satisfied (no comparison data available)
            return {"valid": True, "message": f"Uniqueness rule for field '{field_name}' validated (no comparison data provided)"}
        
        else:
            return {"valid": False, "message": f"Field '{field_name}' not found in data"}
    
    async def _validate_business_logic_rule(self, data: Any, rule: Dict[str, Any], 
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate custom business logic rule"""
        rule_type = rule.get("rule_type")
        
        if not rule_type:
            return {"valid": False, "message": "Business logic rule missing rule_type specification"}
        
        if not isinstance(data, dict):
            return {"valid": False, "message": "Business logic validation requires dictionary data"}
        
        # Age-related business rules
        if rule_type == "minimum_age":
            age_field = rule.get("field", "age")
            min_age = rule.get("minimum", 18)
            
            if age_field in data:
                age = data[age_field]
                if age is not None and isinstance(age, (int, float)) and age < min_age:
                    return {"valid": False, "message": f"Age {age} is below minimum required age of {min_age}"}
                return {"valid": True, "message": f"Age requirement satisfied"}
        
        # Date consistency rules
        elif rule_type == "date_sequence":
            start_field = rule.get("start_field")
            end_field = rule.get("end_field")
            
            if start_field and end_field and start_field in data and end_field in data:
                try:
                    start_date = datetime.fromisoformat(str(data[start_field]).replace('Z', '+00:00'))
                    end_date = datetime.fromisoformat(str(data[end_field]).replace('Z', '+00:00'))
                    
                    if start_date >= end_date:
                        return {"valid": False, "message": f"Start date must be before end date"}
                    return {"valid": True, "message": "Date sequence is valid"}
                except (ValueError, TypeError):
                    return {"valid": False, "message": "Invalid date format for sequence validation"}
        
        # Financial amount rules
        elif rule_type == "positive_amount":
            amount_field = rule.get("field", "amount")
            
            if amount_field in data:
                amount = data[amount_field]
                if amount is not None:
                    try:
                        numeric_amount = float(amount)
                        if numeric_amount <= 0:
                            return {"valid": False, "message": f"Amount must be positive, got {numeric_amount}"}
                        return {"valid": True, "message": "Amount is positive"}
                    except (ValueError, TypeError):
                        return {"valid": False, "message": "Amount must be numeric"}
        
        # Email domain restrictions
        elif rule_type == "allowed_email_domains":
            email_field = rule.get("field", "email")
            allowed_domains = rule.get("allowed_domains", [])
            
            if email_field in data and allowed_domains:
                email = data[email_field]
                if email and isinstance(email, str) and '@' in email:
                    domain = email.split('@')[-1].lower()
                    if domain not in [d.lower() for d in allowed_domains]:
                        return {"valid": False, "message": f"Email domain '{domain}' is not allowed"}
                    return {"valid": True, "message": "Email domain is allowed"}
        
        # Status transition rules
        elif rule_type == "valid_status_transition":
            current_status = data.get("status")
            previous_status = context.get("previous_status")
            allowed_transitions = rule.get("transitions", {})
            
            if previous_status and current_status:
                valid_next_statuses = allowed_transitions.get(previous_status, [])
                if current_status not in valid_next_statuses:
                    return {"valid": False, "message": f"Invalid status transition from '{previous_status}' to '{current_status}'"}
                return {"valid": True, "message": "Status transition is valid"}
        
        # Default fallback
        return {"valid": True, "message": f"Business logic rule '{rule_type}' validated (no specific implementation)"}
    
    # Report generation helpers
    def _generate_business_recommendations(self, results: Dict[str, Any]) -> str:
        """Generate business-focused recommendations based on validation results"""
        recommendations = []
        
        # Analyze validation results for patterns
        total_validations = results.get("validations_performed", 0)
        failed_validations = len(results.get("validation_errors", []))
        success_rate = ((total_validations - failed_validations) / total_validations * 100) if total_validations > 0 else 0
        
        recommendations.append("## Business Recommendations\n")
        
        if success_rate < 50:
            recommendations.append("🚨 **CRITICAL**: Data quality is severely compromised")
            recommendations.append("- Immediate review of data collection processes required")
            recommendations.append("- Implement automated validation at data entry points")
            recommendations.append("- Consider data cleansing initiative")
        elif success_rate < 80:
            recommendations.append("⚠️ **WARNING**: Data quality needs improvement")
            recommendations.append("- Review and strengthen validation rules")
            recommendations.append("- Implement additional quality checks")
            recommendations.append("- Train data entry personnel on quality standards")
        else:
            recommendations.append("✅ **GOOD**: Data quality is acceptable")
            recommendations.append("- Continue current validation practices")
            recommendations.append("- Consider minor optimizations for edge cases")
        
        # Analyze specific error types
        errors = results.get("validation_errors", [])
        error_types = defaultdict(int)
        for error in errors:
            if "format" in error.lower():
                error_types["format"] += 1
            elif "range" in error.lower() or "minimum" in error.lower() or "maximum" in error.lower():
                error_types["range"] += 1
            elif "required" in error.lower() or "missing" in error.lower():
                error_types["completeness"] += 1
            elif "unique" in error.lower():
                error_types["uniqueness"] += 1
        
        if error_types["format"] > 0:
            recommendations.append(f"- Address {error_types['format']} format validation issues")
        if error_types["range"] > 0:
            recommendations.append(f"- Review {error_types['range']} range validation failures")
        if error_types["completeness"] > 0:
            recommendations.append(f"- Improve data completeness ({error_types['completeness']} missing fields)")
        if error_types["uniqueness"] > 0:
            recommendations.append(f"- Investigate {error_types['uniqueness']} uniqueness violations")
        
        recommendations.append(f"\n**Overall Success Rate**: {success_rate:.1f}%\n")
        
        return "\n".join(recommendations)
    
    def _generate_executive_summary(self, results: Dict[str, Any]) -> str:
        """Generate executive summary based on validation results"""
        total_validations = results.get("validations_performed", 0)
        failed_validations = len(results.get("validation_errors", []))
        warnings = len(results.get("warnings", []))
        success_rate = ((total_validations - failed_validations) / total_validations * 100) if total_validations > 0 else 0
        
        summary = []
        summary.append("## Executive Summary\n")
        
        # Overall status
        if success_rate >= 95:
            status = "EXCELLENT"
            status_emoji = "🟢"
        elif success_rate >= 80:
            status = "GOOD"
            status_emoji = "🟡"
        elif success_rate >= 60:
            status = "NEEDS IMPROVEMENT"
            status_emoji = "🟠"
        else:
            status = "CRITICAL"
            status_emoji = "🔴"
        
        summary.append(f"**Data Quality Status**: {status_emoji} {status}")
        summary.append(f"**Success Rate**: {success_rate:.1f}%")
        summary.append(f"**Total Validations**: {total_validations}")
        summary.append(f"**Failed Validations**: {failed_validations}")
        summary.append(f"**Warnings**: {warnings}")
        
        # Key findings
        if failed_validations > 0:
            summary.append("\n**Key Findings**:")
            errors = results.get("validation_errors", [])
            
            # Categorize errors
            critical_errors = [e for e in errors if "critical" in e.lower() or "required" in e.lower()]
            format_errors = [e for e in errors if "format" in e.lower()]
            
            if critical_errors:
                summary.append(f"- {len(critical_errors)} critical data integrity issues")
            if format_errors:
                summary.append(f"- {len(format_errors)} data format inconsistencies")
            if failed_validations - len(critical_errors) - len(format_errors) > 0:
                summary.append(f"- {failed_validations - len(critical_errors) - len(format_errors)} other validation issues")
        
        # Recommendations summary
        if success_rate < 80:
            summary.append("\n**Immediate Actions Required**:")
            summary.append("- Review data quality processes")
            summary.append("- Implement stricter validation controls")
            if failed_validations > 10:
                summary.append("- Consider data cleansing initiative")
        
        compliance_score = results.get("compliance_score", success_rate / 100)
        summary.append(f"\n**Compliance Score**: {compliance_score:.2f}/1.00")
        
        return "\n".join(summary) + "\n"
    
    def _generate_technical_details(self, results: Dict[str, Any]) -> str:
        """Generate detailed technical analysis"""
        details = []
        details.append("## Technical Details\n")
        
        # Validation statistics
        details.append("### Validation Statistics")
        details.append(f"- **Total Records Processed**: {results.get('records_processed', 'N/A')}")
        details.append(f"- **Validation Rules Applied**: {results.get('rules_applied', 'N/A')}")
        details.append(f"- **Processing Time**: {results.get('processing_time', 'N/A')}")
        details.append(f"- **Memory Usage**: {results.get('memory_usage', 'N/A')}")
        
        # Field-level analysis
        if "field_validations" in results:
            details.append("\n### Field-Level Analysis")
            for field, field_results in results["field_validations"].items():
                field_success_rate = field_results.get("success_rate", 0)
                details.append(f"- **{field}**: {field_success_rate:.1f}% success rate")
                
                field_errors = field_results.get("errors", [])
                if field_errors:
                    details.append(f"  - Errors: {len(field_errors)}")
                    for error in field_errors[:3]:  # Show first 3 errors
                        details.append(f"    - {error}")
                    if len(field_errors) > 3:
                        details.append(f"    - ... and {len(field_errors) - 3} more")
        
        # Error breakdown
        if "validation_errors" in results and results["validation_errors"]:
            details.append("\n### Error Breakdown")
            error_categories = defaultdict(list)
            
            for error in results["validation_errors"]:
                if "format" in error.lower():
                    error_categories["Format Errors"].append(error)
                elif "range" in error.lower():
                    error_categories["Range Errors"].append(error)
                elif "required" in error.lower() or "missing" in error.lower():
                    error_categories["Completeness Errors"].append(error)
                elif "unique" in error.lower():
                    error_categories["Uniqueness Errors"].append(error)
                else:
                    error_categories["Other Errors"].append(error)
            
            for category, category_errors in error_categories.items():
                details.append(f"\n**{category}** ({len(category_errors)} issues):")
                for error in category_errors[:5]:  # Show first 5 errors per category
                    details.append(f"- {error}")
                if len(category_errors) > 5:
                    details.append(f"- ... and {len(category_errors) - 5} more")
        
        # Performance metrics
        details.append("\n### Performance Metrics")
        validation_level = results.get("validation_level", "standard")
        details.append(f"- **Validation Level**: {validation_level}")
        details.append(f"- **Custom Validators Applied**: {len(results.get('custom_validators', {}))}")
        details.append(f"- **Schema Version**: {results.get('schema_version', 'N/A')}")
        details.append(f"- **Timestamp**: {results.get('timestamp', datetime.utcnow().isoformat())}")
        
        return "\n".join(details) + "\n"
    
    def _get_pattern_description(self, pattern_name: str) -> str:
        """Get description for validation pattern"""
        descriptions = {
            "email": "Valid email address format",
            "phone": "Valid phone number format",
            "url": "Valid HTTP/HTTPS URL format", 
            "date_iso": "ISO date format (YYYY-MM-DD)",
            "uuid": "Valid UUID format",
            "ipv4": "Valid IPv4 address format"
        }
        return descriptions.get(pattern_name, "Custom validation pattern")
    
    def _get_pattern_examples(self, pattern_name: str) -> List[str]:
        """Get examples for validation pattern"""
        examples = {
            "email": ["user@example.com", "test.email+tag@domain.co.uk"],
            "phone": ["+1-555-123-4567", "(555) 123-4567"],
            "url": ["https://example.com", "https://subdomain.example.org/path"],
            "date_iso": ["2023-12-25", "2024-01-01"],
            "uuid": ["123e4567-e89b-12d3-a456-426614174000"]
        }
        return examples.get(pattern_name, [])
    
    def _get_rule_type_description(self, rule_type: str) -> str:
        """Get description for business rule type"""
        descriptions = {
            "required_field": "Ensures specified fields are present and not null",
            "format_validation": "Validates data format against patterns",
            "range_check": "Ensures values fall within specified ranges",
            "dependency_check": "Validates field dependencies and relationships",
            "uniqueness_check": "Ensures values are unique within dataset",
            "referential_integrity": "Validates foreign key relationships",
            "business_logic": "Custom business rule validation"
        }
        return descriptions.get(rule_type, "Custom validation rule")


# Singleton instance
mcp_validation_tools = MCPValidationTools()