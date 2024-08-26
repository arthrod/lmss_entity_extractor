import json
import logging
import argparse
from pathlib import Path
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List
from app.entity_extraction import EntityExtractor, ExtractedEntity

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

extractor = EntityExtractor()

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

def save_results(data: List[ExtractedEntity], file_path: str):
    logger.info(f"Saving results to {file_path}")
    with open(file_path, "w") as f:
        json.dump([entity.dict() for entity in data], f, indent=2)

def process_text(text: str) -> List[ExtractedEntity]:
    logger.info("Extracting entities")
    extracted_entities = extractor.extract_entities(text)
    return extracted_entities

def entity_to_output(entity: ExtractedEntity) -> EntityOutput:
    return EntityOutput(
        text=entity.text,
        start=entity.start,
        end=entity.end,
        type=entity.type,
        label=getattr(entity, 'label', None),
        confidence=getattr(entity, 'confidence', None),
        source=getattr(entity, 'source', None)
    )

def print_summary(entities: List[ExtractedEntity]):
    print("\nExtraction Summary:")
    print(f"Total entities extracted: {len(entities)}")

    print("\nSample Extracted Entities:")
    for entity in entities[:5]:  # Print first 5 entities
        print(f"- {entity.text} ({entity.type})")

def calculate_statistics(entities: List[ExtractedEntity]) -> dict:
    stats = {
        "total_entities": len(entities),
        "entity_types": {},
    }

    for entity in entities:
        if entity.type not in stats["entity_types"]:
            stats["entity_types"][entity.type] = 0
        stats["entity_types"][entity.type] += 1

    return stats

@app.post("/extract", response_model=List[EntityOutput])
async def extract_entities(document: DocumentInput):
    try:
        results = process_text(document.text)
        return [entity_to_output(entity) for entity in results]
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing text")

@app.post("/run_gui")
async def run_gui(background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(main)
        return {"message": "GUI functionality started in the background"}
    except Exception as e:
        logger.error(f"Error starting GUI functionality: {str(e)}")
        raise HTTPException(status_code=500, detail="Error starting GUI functionality")

def main():
    parser = argparse.ArgumentParser(description="Extract entities from text")
    parser.add_argument("--input", type=str, help="Path to input text file")
    parser.add_argument(
        "--output",
        type=str,
        default="app/lmss/extraction_results.json",
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--stats",
        type=str,
        default="app/lmss/extraction_stats.json",
        help="Path to output statistics JSON file",
    )
    args = parser.parse_args()

    if args.input:
        with open(args.input, "r") as f:
            text = f.read()
    else:
        # Example text if no input file is provided
        text = """
        The intellectual property lawyer specializes in patent law and copyright infringement cases.
        She also handles trademark disputes and trade secret litigation. Recently, she's been working
        on a high-profile case involving software licensing and open source compliance in Paris, Texas.
        """

    results = process_text(text)
    save_results(results, args.output)
    print_summary(results)

    # Calculate and save statistics
    stats = calculate_statistics(results)
    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Full results saved to {args.output}")
    logger.info(f"Statistics saved to {args.stats}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)