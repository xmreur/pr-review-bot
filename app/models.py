from pydantic import BaseModel
from typing import Literal


class Finding(BaseModel):
    path: str
    line: int
    side: Literal["RIGHT", "LEFT"] = "RIGHT"
    severity: Literal['low', 'medium', 'high', 'critical']
    title: str
    body: str
    # Optional evidence fields used during the verification pass.
    # Extra fields from the LLM are ignored on parse unless declared,
    # so keep these optional with safe defaults.
    confidence: Literal['low', 'medium', 'high'] | None = None
    evidence: str | None = None


class Review(BaseModel):
    findings: list[Finding]