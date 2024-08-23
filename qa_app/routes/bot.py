from fastapi import APIRouter
from qa_app.models.bot import QaBotResponse, QaBotRequest
from qa_app.helpers.qa_generator import QaGenerator

router = APIRouter()


@router.post("/qa", response_model=QaBotResponse)
async def fetch_answers(request: QaBotRequest):
    return QaGenerator().execute(request.document_path, request.document_type, request.questions)
