from typing import List, Dict

from pydantic import BaseModel, field_validator

SUPPORTED_DOCUMENT_TYPES = ["pdf", "json"]


class QaResponse(BaseModel):
    question: str
    answer: str


class QaBotResponse(BaseModel):
    result: Dict[str, str]


class QaBotRequest(BaseModel):
    document_path: str
    document_type: str
    questions: List[str]

    @field_validator('document_type')
    @classmethod
    def validate_document_type(cls, document_type: str) -> str:
        if document_type not in SUPPORTED_DOCUMENT_TYPES:
            raise ValueError('INVALID DOCUMENT TYPE')
        return document_type
