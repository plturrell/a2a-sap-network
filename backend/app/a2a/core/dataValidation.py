"""
Data Validation and Transformation Verification
"""

import json
from typing import Dict, List, Any, Optional
import logging
from jsonschema import validate, ValidationError
import hashlib

logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of validation operation"""

    def __init__(
        self,
        is_valid: bool,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
    ):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.metadata = {}

    def add_error(self, error: str):
        """Add validation error"""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str):
        """Add validation warning"""
        self.warnings.append(warning)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


class DataValidator:
    """Validate data for catalog operations"""

    def __init__(self):
        self.schemas = self._load_schemas()

    def _load_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load validation schemas"""
        return {
            "catalog_metadata": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "description": {"type": "string"},
                    "type": {"type": "string"},
                    "owner": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "columns": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["name", "type"],
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "nullable": {"type": "boolean"},
                                "description": {"type": "string"},
                            },
                        },
                    },
                },
            },
            "standardized_entity": {
                "type": "object",
                "required": ["original", "standardized", "confidence"],
                "properties": {
                    "original": {"type": "object"},
                    "standardized": {"type": "object"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "metadata": {"type": "object"},
                },
            },
        }

    def validate_catalog_metadata(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate catalog metadata"""
        result = ValidationResult(True)

        try:
            validate(instance=data, schema=self.schemas["catalog_metadata"])
        except ValidationError as e:
            result.add_error(f"Schema validation failed: {e.message}")
            return result

        # Additional business logic validation
        if "columns" in data:
            column_names = [col["name"] for col in data["columns"]]
            if len(column_names) != len(set(column_names)):
                result.add_error("Duplicate column names found")

        # Validate data types
        if "columns" in data:
            valid_types = {
                "string",
                "integer",
                "long",
                "float",
                "double",
                "boolean",
                "timestamp",
                "date",
                "binary",
                "decimal",
            }
            for col in data["columns"]:
                if col["type"].lower() not in valid_types:
                    result.add_warning(
                        f"Non-standard type '{col['type']}' for column '{col['name']}'"
                    )

        return result

    def validate_standardized_data(
        self, original_data: List[Dict[str, Any]], standardized_data: List[Dict[str, Any]]
    ) -> ValidationResult:
        """Validate standardization transformation"""
        result = ValidationResult(True)

        # Check record counts
        if len(original_data) != len(standardized_data):
            result.add_error(
                f"Record count mismatch: original={len(original_data)}, "
                f"standardized={len(standardized_data)}"
            )
            return result

        # Validate each standardized record
        for i, (_, std) in enumerate(zip(original_data, standardized_data)):
            # Check structure
            if "standardized" not in std:
                result.add_error(f"Record {i}: Missing 'standardized' field")
                continue

            # Validate confidence scores
            if "confidence" in std:
                conf = std["confidence"]
                if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
                    result.add_error(f"Record {i}: Invalid confidence score: {conf}")

            # Check for data loss
            if "error" in std and std.get("standardized") is False:
                result.add_warning(f"Record {i}: Standardization failed - {std.get('error')}")

        # Calculate success rate
        successful = sum(1 for r in standardized_data if r.get("standardized") != False)
        success_rate = successful / len(standardized_data) if standardized_data else 0

        result.metadata["success_rate"] = success_rate
        result.metadata["total_records"] = len(standardized_data)
        result.metadata["successful_records"] = successful

        if success_rate < 0.8:  # Less than 80% success
            result.add_warning(f"Low standardization success rate: {success_rate:.2%}")

        return result


class TransformationValidator:
    """Validate data transformations between formats"""

    def validate_ord_document(self, ord_doc: Dict[str, Any]) -> ValidationResult:
        """Validate ORD document structure"""
        result = ValidationResult(True)

        # Check required fields
        required_fields = [
            "openResourceDiscovery",
            "products",
            "packages",
            "apis",
            "events",
            "entityTypes",
        ]

        for field in required_fields:
            if field not in ord_doc:
                result.add_error(f"Missing required ORD field: {field}")

        # Validate ORD version
        if "openResourceDiscovery" in ord_doc:
            version = ord_doc["openResourceDiscovery"]
            if not isinstance(version, str) or not version.startswith("1."):
                result.add_error(f"Invalid ORD version: {version}")

        # Validate products
        if "products" in ord_doc and isinstance(ord_doc["products"], list):
            for i, product in enumerate(ord_doc["products"]):
                if "ordId" not in product:
                    result.add_error(f"Product {i}: Missing ordId")
                if "title" not in product:
                    result.add_error(f"Product {i}: Missing title")

        # Validate APIs
        if "apis" in ord_doc and isinstance(ord_doc["apis"], list):
            for i, api in enumerate(ord_doc["apis"]):
                self._validate_ord_api(api, i, result)

        return result

    def _validate_ord_api(self, api: Dict[str, Any], index: int, result: ValidationResult):
        """Validate ORD API resource"""
        required = ["ordId", "title", "visibility", "releaseStatus"]

        for field in required:
            if field not in api:
                result.add_error(f"API {index}: Missing required field '{field}'")

        # Validate visibility
        if "visibility" in api and api["visibility"] not in ["public", "internal", "private"]:
            result.add_error(f"API {index}: Invalid visibility '{api['visibility']}'")

        # Validate release status
        if "releaseStatus" in api and api["releaseStatus"] not in ["active", "beta", "deprecated"]:
            result.add_warning(f"API {index}: Non-standard release status '{api['releaseStatus']}'")

    def validate_csn_document(self, csn_doc: Dict[str, Any]) -> ValidationResult:
        """Validate CSN document structure"""
        result = ValidationResult(True)

        # Check required fields
        if "definitions" not in csn_doc:
            result.add_error("Missing 'definitions' in CSN document")
            return result

        # Validate version
        if "$version" in csn_doc:
            version = csn_doc["$version"]
            if not isinstance(version, str):
                result.add_error(f"Invalid CSN version type: {type(version)}")

        # Validate definitions
        for name, definition in csn_doc.get("definitions", {}).items():
            self._validate_csn_definition(name, definition, result)

        return result

    def _validate_csn_definition(
        self, name: str, definition: Dict[str, Any], result: ValidationResult
    ):
        """Validate CSN definition"""
        if "kind" not in definition:
            result.add_error(f"Definition '{name}': Missing 'kind'")
            return

        kind = definition["kind"]
        if kind not in ["entity", "type", "service", "context"]:
            result.add_warning(f"Definition '{name}': Unknown kind '{kind}'")

        # Validate entity
        if kind == "entity" and "elements" in definition:
            elements = definition["elements"]
            if not isinstance(elements, dict):
                result.add_error(f"Entity '{name}': 'elements' must be an object")
            else:
                for elem_name, elem_def in elements.items():
                    if "type" not in elem_def:
                        result.add_error(f"Entity '{name}', element '{elem_name}': Missing type")


class DataIntegrityChecker:
    """Check data integrity for transformations"""

    def calculate_checksum(self, data: Any) -> str:
        """Calculate SHA256 checksum for data"""
        if isinstance(data, dict) or isinstance(data, list):
            # Convert to stable JSON string
            json_str = json.dumps(data, sort_keys=True, ensure_ascii=True)
            data_bytes = json_str.encode("utf-8")
        elif isinstance(data, str):
            data_bytes = data.encode("utf-8")
        else:
            data_bytes = str(data).encode("utf-8")

        return hashlib.sha256(data_bytes).hexdigest()

    def verify_transformation_integrity(
        self,
        source_data: Dict[str, Any],
        transformed_data: Dict[str, Any],
        expected_fields: List[str],
    ) -> ValidationResult:
        """Verify data integrity after transformation"""
        result = ValidationResult(True)

        # Check critical fields preserved
        for field in expected_fields:
            if field in source_data and field not in transformed_data:
                result.add_error(f"Critical field '{field}' lost in transformation")

        # Check for data corruption
        if "checksum" in source_data:
            # Verify specific fields that should be preserved
            preserved_fields = ["id", "name", "type"]
            for field in preserved_fields:
                if field in source_data:
                    source_value = source_data[field]
                    transformed_value = transformed_data.get(field)

                    if source_value != transformed_value:
                        result.add_warning(
                            f"Field '{field}' changed: '{source_value}' -> '{transformed_value}'"
                        )

        # Add checksums
        result.metadata["source_checksum"] = self.calculate_checksum(source_data)
        result.metadata["transformed_checksum"] = self.calculate_checksum(transformed_data)

        return result


# Global validator instances
_data_validator = None
_transform_validator = None
_integrity_checker = None


def get_data_validator() -> DataValidator:
    """Get global data validator"""
    global _data_validator
    if _data_validator is None:
        _data_validator = DataValidator()
    return _data_validator


def get_transform_validator() -> TransformationValidator:
    """Get global transformation validator"""
    global _transform_validator
    if _transform_validator is None:
        _transform_validator = TransformationValidator()
    return _transform_validator


def get_integrity_checker() -> DataIntegrityChecker:
    """Get global integrity checker"""
    global _integrity_checker
    if _integrity_checker is None:
        _integrity_checker = DataIntegrityChecker()
    return _integrity_checker
