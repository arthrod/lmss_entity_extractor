from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from app.entity_extraction import EntityExtractor
from app.run_extraction import process_text, calculate_statistics

app = FastAPI()

class DocumentInput(BaseModel):
    text: str

class EntityOutput(BaseModel):
    text: str
    start: int
    end: int
    type: str
    label: str = None
    confidence: float = None
    source: str = None

@app.post("/api/extract_entities", response_model=List[EntityOutput])
async def extract_entities(document: DocumentInput):
    try:
        entities = process_text(document.text)
        return entities
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/api/extract_entities_with_stats")
async def extract_entities_with_stats(document: DocumentInput):
    try:
        entities = process_text(document.text)
        stats = calculate_statistics(entities)
        return {"entities": entities, "statistics": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)