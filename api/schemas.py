from pydantic import BaseModel, Field


class TicketRequest(BaseModel):

    ticket: str = Field(
        ...,
        example="I was charged twice for the same order."
    )


class TicketResponse(BaseModel):

    category: str

    subcategory: str

    priority: str

    sentiment: str

    summary: str