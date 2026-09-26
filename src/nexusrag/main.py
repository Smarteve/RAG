from fastapi import FastAPI,HTTPException,UploadFile
from pydantic import BaseModel
from nexusrag.schemas.document import (
    MAX_DOCUMENT_SIZE_BYTES,
    DocumentMetadata,
)

app = FastAPI(title="NexusRAG")


class HealthCheckResponse(BaseModel):
    status: str = "ok"


@app.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    return HealthCheckResponse(status="ok")

@app.get("/document", response_model=DocumentMetadata)
async def upload_document(file:UploadFile) -> DocumentMetadata:
    if file.content_type != 'application/pdf':
        raise HTTPException(415)

    if not file.name:
        raise HTTPException()


    try:
        total_size_bytes = 0
        chunk_size_bytes = 1024*1024

        while True:
            chunk= await file.read(chunk_size_bytes)
            if not chunk: 
                break
            total_size_bytes+=chunk_size_bytes
            if total_size_bytes > MAX_DOCUMENT_SIZE_BYTES:
                raise HTTPException()
            return DocumentMetadata(filename=file.filename,media_type=file.content_type,size_bytes=total_size_bytes)

    except ValidationError as error:
        raise HTTPException() from error


    finally:
        await file.close()



    

