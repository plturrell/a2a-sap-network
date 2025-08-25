"""
Pagination utilities for efficient data retrieval
"""

from typing import Dict, Any, List, Optional, TypeVar, Generic
from dataclasses import dataclass
import math

T = TypeVar('T')

@dataclass
class PaginationParams:
    """Pagination parameters"""
    page: int = 1
    page_size: int = 50
    max_page_size: int = 1000

    def __post_init__(self):
        # Validate and sanitize
        self.page = max(1, self.page)
        self.page_size = min(max(1, self.page_size), self.max_page_size)

    @property
    def offset(self) -> int:
        """Calculate offset for database queries"""
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        """Get limit for database queries"""
        return self.page_size


@dataclass
class PaginatedResponse(Generic[T]):
    """Generic paginated response"""
    items: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool

    @classmethod
    def create(cls, items: List[T], total: int, params: PaginationParams):
        """Create paginated response from items and params"""
        total_pages = math.ceil(total / params.page_size)
        return cls(
            items=items,
            total=total,
            page=params.page,
            page_size=params.page_size,
            total_pages=total_pages,
            has_next=params.page < total_pages,
            has_previous=params.page > 1
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "items": self.items,
            "pagination": {
                "total": self.total,
                "page": self.page,
                "page_size": self.page_size,
                "total_pages": self.total_pages,
                "has_next": self.has_next,
                "has_previous": self.has_previous
            }
        }


class CursorPagination:
    """Cursor-based pagination for large datasets"""

    @staticmethod
    def encode_cursor(value: Any) -> str:
        """Encode cursor value"""
        import base64
        import json
        return base64.b64encode(json.dumps(value).encode()).decode()

    @staticmethod
    def decode_cursor(cursor: str) -> Any:
        """Decode cursor value"""
        import base64
        import json
        try:
            return json.loads(base64.b64decode(cursor.encode()).decode())
        except:
            return None

    @dataclass
    class CursorParams:
        """Cursor pagination parameters"""
        cursor: Optional[str] = None
        limit: int = 50
        direction: str = "next"  # "next" or "previous"

        def get_decoded_cursor(self) -> Any:
            """Get decoded cursor value"""
            if not self.cursor:
                return None
            return CursorPagination.decode_cursor(self.cursor)

    @dataclass
    class CursorResponse(Generic[T]):
        """Cursor-based pagination response"""
        items: List[T]
        next_cursor: Optional[str]
        previous_cursor: Optional[str]
        has_more: bool

        def to_dict(self) -> Dict[str, Any]:
            """Convert to dictionary"""
            return {
                "items": self.items,
                "cursors": {
                    "next": self.next_cursor,
                    "previous": self.previous_cursor
                },
                "has_more": self.has_more
            }


def paginate_query(query: str, params: PaginationParams) -> str:
    """Add pagination to SQL query"""
    return f"{query} LIMIT {params.limit} OFFSET {params.offset}"


def calculate_pagination_stats(total_items: int, params: PaginationParams) -> Dict[str, Any]:
    """Calculate pagination statistics"""
    total_pages = math.ceil(total_items / params.page_size)

    return {
        "current_page": params.page,
        "page_size": params.page_size,
        "total_items": total_items,
        "total_pages": total_pages,
        "has_next": params.page < total_pages,
        "has_previous": params.page > 1,
        "start_index": params.offset + 1,
        "end_index": min(params.offset + params.page_size, total_items)
    }
