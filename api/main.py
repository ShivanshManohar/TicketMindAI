from fastapi import FastAPI

from schemas import TicketRequest, TicketResponse
from inference import predict_ticket
from logger import logger

app = FastAPI(
    title="TicketMindAI",
    description="""
AI-powered customer support ticket classification API.

This API analyzes customer support tickets and returns structured metadata including:

- Category
- Subcategory
- Priority
- Sentiment
- Summary
""",
    version="1.0.0",
    contact={
        "name": "Shivansh Mehra",
    },
)


@app.get(
    "/",
    tags=["Health"]
)
def health():
    return {
        "status": "healthy",
        "service": "TicketMindAI",
        "version": "1.0.0"
    }


@app.get(
    "/version",
    tags=["Health"]
)
def version():
    return {
        "model": "Llama 3.2 + LoRA",
        "framework": "FastAPI",
        "version": "1.0.0"
    }


@app.post(
    "/predict",
    response_model=TicketResponse,
    tags=["Prediction"],
    summary="Predict ticket metadata",
    description="Analyze a customer support ticket and return structured metadata."
)
def predict(request: TicketRequest):

    logger.info("Prediction request received.")

    result = predict_ticket(request.ticket)

    logger.info("Prediction completed successfully.")

    return result