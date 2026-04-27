from typing import Optional

from pydantic import BaseModel, Field

from rust_assistant.domain.enums import Crate, ItemType


class SearchFilters(BaseModel):
    """Optional filters accepted by the search endpoint."""

    crate: Optional[Crate] = None
    item_type: Optional[ItemType] = None


class SearchHit(BaseModel):
    """One search hit returned to the API client."""

    title: str
    source_path: str
    section: Optional[str] = None
    item_path: Optional[str] = None
    crate: Optional[str] = None
    item_type: Optional[str] = None
    score: float = Field(..., ge=0.0, le=1.0)
    snippet: str