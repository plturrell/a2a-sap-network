"""
A2A Message Serialization and Deserialization Protocols
Provides standardized serialization for A2A messages with compression and validation
"""

import json
import pickle
import gzip
import base64
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum
import logging
from dataclasses import dataclass

from .a2aTypes import A2AMessage, MessagePart, MessageRole

logger = logging.getLogger(__name__)


class SerializationFormat(str, Enum):
    JSON = "json"
    JSON_COMPRESSED = "json_gzip"
    BINARY = "binary"
    BINARY_COMPRESSED = "binary_gzip"


class MessageSerializationError(Exception):
    """Exception raised during message serialization/deserialization"""


@dataclass
class SerializedMessage:
    """Wrapper for serialized message with metadata"""

    data: bytes
    format: SerializationFormat
    checksum: str
    size_bytes: int
    compression_ratio: Optional[float] = None
    serialized_at: datetime = None

    def __post_init__(self):
        if self.serialized_at is None:
            self.serialized_at = datetime.utcnow()


class A2AMessageSerializer:
    """A2A Protocol compliant message serializer with compression and validation"""

    def __init__(
        self,
        compression_threshold: int = 1024,
        default_format: SerializationFormat = SerializationFormat.JSON,
    ):
        """
        Initialize serializer

        Args:
            compression_threshold: Messages larger than this (bytes) will be compressed
            default_format: Default serialization format to use
        """
        self.compression_threshold = compression_threshold
        self.default_format = default_format

        # Statistics
        self.serialization_stats = {
            "messages_serialized": 0,
            "messages_deserialized": 0,
            "total_bytes_serialized": 0,
            "total_bytes_compressed": 0,
            "compression_savings": 0,
            "errors": 0,
        }

    def serialize(
        self,
        message: A2AMessage,
        serialization_format: Optional[SerializationFormat] = None,
        force_compression: bool = False,
    ) -> SerializedMessage:
        """
        Serialize A2A message to specified format

        Args:
            message: A2A message to serialize
            format: Serialization format (uses default if None)
            force_compression: Force compression even if below threshold

        Returns:
            SerializedMessage with serialized data and metadata
        """
        try:
            serialization_format = serialization_format or self.default_format

            # Convert message to dictionary
            message_dict = self._message_to_dict(message)

            # Validate message structure
            self._validate_message_dict(message_dict)

            # Choose serialization method
            if serialization_format in [SerializationFormat.JSON, SerializationFormat.JSON_COMPRESSED]:
                raw_data = json.dumps(
                    message_dict, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
            elif serialization_format in [SerializationFormat.BINARY, SerializationFormat.BINARY_COMPRESSED]:
                raw_data = pickle.dumps(message_dict, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                raise MessageSerializationError(f"Unsupported serialization format: {serialization_format}")

            original_size = len(raw_data)

            # Apply compression if needed
            if (original_size > self.compression_threshold or force_compression) and serialization_format in [
                SerializationFormat.JSON_COMPRESSED,
                SerializationFormat.BINARY_COMPRESSED,
            ]:

                compressed_data = gzip.compress(raw_data, compresslevel=6)
                final_data = compressed_data
                compression_ratio = len(compressed_data) / original_size

                logger.debug(
                    f"Compressed message from {original_size} to {len(compressed_data)} bytes "
                    f"(ratio: {compression_ratio:.2f})"
                )

                self.serialization_stats["total_bytes_compressed"] += original_size - len(
                    compressed_data
                )

            else:
                final_data = raw_data
                compression_ratio = None

                # Switch to non-compressed format if compression not applied
                if serialization_format == SerializationFormat.JSON_COMPRESSED:
                    serialization_format = SerializationFormat.JSON
                elif serialization_format == SerializationFormat.BINARY_COMPRESSED:
                    serialization_format = SerializationFormat.BINARY

            # Calculate checksum
            checksum = hashlib.sha256(final_data).hexdigest()

            # Update statistics
            self.serialization_stats["messages_serialized"] += 1
            self.serialization_stats["total_bytes_serialized"] += len(final_data)

            return SerializedMessage(
                data=final_data,
                format=serialization_format,
                checksum=checksum,
                size_bytes=len(final_data),
                compression_ratio=compression_ratio,
            )

        except Exception as e:
            self.serialization_stats["errors"] += 1
            logger.error(f"Message serialization failed: {e}")
            raise MessageSerializationError(f"Serialization failed: {e}")

    def deserialize(self, serialized_message: SerializedMessage) -> A2AMessage:
        """
        Deserialize A2A message from SerializedMessage

        Args:
            serialized_message: Serialized message to deserialize

        Returns:
            A2AMessage instance
        """
        try:
            # Verify checksum
            actual_checksum = hashlib.sha256(serialized_message.data).hexdigest()
            if actual_checksum != serialized_message.checksum:
                raise MessageSerializationError(
                    "Checksum verification failed - message may be corrupted"
                )

            # Decompress if needed
            if serialized_message.format in [
                SerializationFormat.JSON_COMPRESSED,
                SerializationFormat.BINARY_COMPRESSED,
            ]:
                try:
                    raw_data = gzip.decompress(serialized_message.data)
                except Exception as e:
                    raise MessageSerializationError(f"Decompression failed: {e}")
            else:
                raw_data = serialized_message.data

            # Deserialize based on format
            if serialized_message.format in [
                SerializationFormat.JSON,
                SerializationFormat.JSON_COMPRESSED,
            ]:
                try:
                    message_dict = json.loads(raw_data.decode("utf-8"))
                except json.JSONDecodeError as e:
                    raise MessageSerializationError(f"JSON deserialization failed: {e}")

            elif serialized_message.format in [
                SerializationFormat.BINARY,
                SerializationFormat.BINARY_COMPRESSED,
            ]:
                try:
                    message_dict = pickle.loads(raw_data)
                except Exception as e:
                    raise MessageSerializationError(f"Binary deserialization failed: {e}")
            else:
                raise MessageSerializationError(
                    f"Unsupported format for deserialization: {serialized_message.format}"
                )

            # Validate deserialized structure
            self._validate_message_dict(message_dict)

            # Convert back to A2AMessage
            message = self._dict_to_message(message_dict)

            # Update statistics
            self.serialization_stats["messages_deserialized"] += 1

            return message

        except Exception as e:
            self.serialization_stats["errors"] += 1
            logger.error(f"Message deserialization failed: {e}")
            raise MessageSerializationError(f"Deserialization failed: {e}")

    def serialize_to_base64(
        self, message: A2AMessage, format: Optional[SerializationFormat] = None
    ) -> str:
        """
        Serialize message and encode as base64 string for transport

        Args:
            message: A2A message to serialize
            format: Serialization format

        Returns:
            Base64 encoded serialized message with metadata header
        """
        try:
            serialized = self.serialize(message, format)

            # Create metadata header
            metadata = {
                "format": serialized.format.value,
                "checksum": serialized.checksum,
                "size": serialized.size_bytes,
                "compression_ratio": serialized.compression_ratio,
                "serialized_at": serialized.serialized_at.isoformat(),
            }

            # Encode metadata and data
            metadata_json = json.dumps(metadata, separators=(",", ":"))
            metadata_b64 = base64.b64encode(metadata_json.encode("utf-8")).decode("ascii")
            data_b64 = base64.b64encode(serialized.data).decode("ascii")

            # Format: metadata_length:metadata_base64:data_base64
            return f"{len(metadata_b64)}:{metadata_b64}:{data_b64}"

        except Exception as e:
            logger.error(f"Base64 serialization failed: {e}")
            raise MessageSerializationError(f"Base64 serialization failed: {e}")

    def deserialize_from_base64(self, base64_message: str) -> A2AMessage:
        """
        Deserialize message from base64 encoded string

        Args:
            base64_message: Base64 encoded message with metadata

        Returns:
            A2AMessage instance
        """
        try:
            # Parse format: metadata_length:metadata_base64:data_base64
            parts = base64_message.split(":", 2)
            if len(parts) != 3:
                raise MessageSerializationError("Invalid base64 message format")

            metadata_length = int(parts[0])
            metadata_b64 = parts[1]
            data_b64 = parts[2]

            # Verify metadata length
            if len(metadata_b64) != metadata_length:
                raise MessageSerializationError("Metadata length mismatch")

            # Decode metadata
            metadata_json = base64.b64decode(metadata_b64).decode("utf-8")
            metadata = json.loads(metadata_json)

            # Decode data
            data = base64.b64decode(data_b64)

            # Reconstruct SerializedMessage
            serialized_message = SerializedMessage(
                data=data,
                format=SerializationFormat(metadata["format"]),
                checksum=metadata["checksum"],
                size_bytes=metadata["size"],
                compression_ratio=metadata.get("compression_ratio"),
                serialized_at=datetime.fromisoformat(metadata["serialized_at"]),
            )

            return self.deserialize(serialized_message)

        except Exception as e:
            logger.error(f"Base64 deserialization failed: {e}")
            raise MessageSerializationError(f"Base64 deserialization failed: {e}")

    def _message_to_dict(self, message: A2AMessage) -> Dict[str, Any]:
        """Convert A2AMessage to dictionary"""
        return {
            "messageId": message.messageId,
            "role": message.role.value,
            "parts": [
                {"kind": part.kind, "text": part.text, "data": part.data, "file": part.file}
                for part in message.parts
            ],
            "taskId": message.taskId,
            "contextId": message.contextId,
            "timestamp": message.timestamp,
            "signature": message.signature,
            "_protocol_version": "2.9",
            "_serialization_version": "1.0",
        }

    def _dict_to_message(self, message_dict: Dict[str, Any]) -> A2AMessage:
        """Convert dictionary to A2AMessage"""

        # Convert parts
        parts = []
        for part_dict in message_dict.get("parts", []):
            parts.append(
                MessagePart(
                    kind=part_dict.get("kind", "text"),
                    text=part_dict.get("text"),
                    data=part_dict.get("data"),
                    file=part_dict.get("file"),
                )
            )

        return A2AMessage(
            messageId=message_dict["messageId"],
            role=MessageRole(message_dict["role"]),
            parts=parts,
            taskId=message_dict.get("taskId"),
            contextId=message_dict.get("contextId"),
            timestamp=message_dict.get("timestamp"),
            signature=message_dict.get("signature"),
        )

    def _validate_message_dict(self, message_dict: Dict[str, Any]) -> None:
        """Validate message dictionary structure"""
        required_fields = ["messageId", "role", "parts"]

        for field in required_fields:
            if field not in message_dict:
                raise MessageSerializationError(f"Missing required field: {field}")

        # Validate role
        try:
            MessageRole(message_dict["role"])
        except ValueError:
            raise MessageSerializationError(f"Invalid message role: {message_dict['role']}")

        # Validate parts
        if not isinstance(message_dict["parts"], list):
            raise MessageSerializationError("Parts must be a list")

        for i, part in enumerate(message_dict["parts"]):
            if not isinstance(part, dict) or "kind" not in part:
                raise MessageSerializationError(f"Invalid part at index {i}")

    def get_optimal_format(self, message: A2AMessage) -> SerializationFormat:
        """
        Determine optimal serialization format for message

        Args:
            message: Message to analyze

        Returns:
            Recommended SerializationFormat
        """
        # Quick size estimation
        estimated_size = (
            len(str(message.messageId))
            + len(message.role.value)
            + sum(len(str(part.text or "")) + len(str(part.data or "")) for part in message.parts)
        )

        # For large messages with data, prefer binary compression
        has_binary_data = any(
            part.data and isinstance(part.data, (bytes, bytearray)) for part in message.parts
        )
        has_large_data = any(part.data and len(str(part.data)) > 1000 for part in message.parts)

        if has_binary_data or (has_large_data and estimated_size > self.compression_threshold):
            return SerializationFormat.BINARY_COMPRESSED
        elif estimated_size > self.compression_threshold:
            return SerializationFormat.JSON_COMPRESSED
        elif has_binary_data:
            return SerializationFormat.BINARY
        else:
            return SerializationFormat.JSON

    def get_statistics(self) -> Dict[str, Any]:
        """Get serialization statistics"""
        stats = self.serialization_stats.copy()

        if stats["total_bytes_serialized"] > 0:
            stats["average_message_size"] = stats["total_bytes_serialized"] / max(
                stats["messages_serialized"], 1
            )
            stats["compression_efficiency"] = stats["total_bytes_compressed"] / max(
                stats["total_bytes_serialized"], 1
            )
        else:
            stats["average_message_size"] = 0
            stats["compression_efficiency"] = 0

        return stats

    def reset_statistics(self) -> None:
        """Reset serialization statistics"""
        for key in self.serialization_stats:
            self.serialization_stats[key] = 0


# Global serializer instance
_default_serializer = None


def get_message_serializer() -> A2AMessageSerializer:
    """Get global message serializer instance"""
    global _default_serializer

    if _default_serializer is None:
        _default_serializer = A2AMessageSerializer()

    return _default_serializer


# Convenience functions
def serialize_message(
    message: A2AMessage, format: Optional[SerializationFormat] = None
) -> SerializedMessage:
    """Serialize A2A message using global serializer"""
    return get_message_serializer().serialize(message, format)


def deserialize_message(serialized_message: SerializedMessage) -> A2AMessage:
    """Deserialize A2A message using global serializer"""
    return get_message_serializer().deserialize(serialized_message)


def serialize_to_base64(message: A2AMessage, format: Optional[SerializationFormat] = None) -> str:
    """Serialize message to base64 string using global serializer"""
    return get_message_serializer().serialize_to_base64(message, format)


def deserialize_from_base64(base64_message: str) -> A2AMessage:
    """Deserialize message from base64 string using global serializer"""
    return get_message_serializer().deserialize_from_base64(base64_message)
