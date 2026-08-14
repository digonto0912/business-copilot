from typing import List, Literal

from pydantic import BaseModel, Field


class QAPair(BaseModel):
    question_asked: str
    user_answer: str
    verification: Literal[
        "verified",
        "uncertain_match",
    ]


class UnpromptedContext(BaseModel):
    raw_text: str
    topic_hint: str


class VerifiedFacts(BaseModel):
    qa_pairs: List[QAPair] = Field(
        default_factory=list
    )

    unprompted_context: List[UnpromptedContext] = Field(
        default_factory=list
    )