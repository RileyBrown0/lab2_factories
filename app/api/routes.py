from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from app.services.email_topic_inference import EmailTopicInferenceService
from app.dataclasses import Email
 
import json
import uuid
from pathlib import Path
 
router = APIRouter()
 
datadir = Path("data")
TOPICS_FILE = datadir / "topic_keywords.json"
EMAILS_FILE = datadir / "stored_emails.json"
 
 
def _load_topics() -> dict:
    with open(TOPICS_FILE) as f:
        return json.load(f)
 
def _save_topics(topics: dict) -> None:
    with open(TOPICS_FILE, "w") as f:
        json.dump(topics, f, indent=2)
 
def _load_emails() -> list:
    if not EMAILS_FILE.exists():
        return []
    with open(EMAILS_FILE) as f:
        return json.load(f)
 
def _save_emails(emails: list) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    with open(EMAILS_FILE, "w") as f:
        json.dump(emails, f, indent=2)
 

class EmailRequest(BaseModel):
    subject: str
    body: str
 
class EmailWithTopicRequest(BaseModel):
    subject: str
    body: str
    topic: str
 
class EmailClassificationResponse(BaseModel):
    predicted_topic: str
    topic_scores: Dict[str, float]
    features: Dict[str, Any]
    available_topics: List[str]
 
class EmailAddResponse(BaseModel):
    message: str
    email_id: int
 

 
class TopicCreateRequest(BaseModel):
    name: str
    description: str
    keywords: List[str]
 
class TopicResponse(BaseModel):
    name: str
    description: str
    keywords: List[str]
 

 
class StoredEmailRequest(BaseModel):
    subject: str
    body: str
    ground_truth: Optional[str] = None   # optional label for similarity classifier
 
class StoredEmailResponse(BaseModel):
    id: str
    subject: str
    body: str
    ground_truth: Optional[str]
 

 
class ClassifyRequest(BaseModel):
    subject: str
    body: str
    mode: str = "topic"   # "topic" (default) or "email" (nearest stored email)
 

 
@router.post("/emails/classify", response_model=EmailClassificationResponse)
async def classify_email(request: ClassifyRequest):   # Q3: now uses ClassifyRequest
    try:
        inference_service = EmailTopicInferenceService()
        email = Email(subject=request.subject, body=request.body)
 
        # Q3: if mode="email", find the most similar stored labelled email
        if request.mode == "email":
            stored = _load_emails()
            labelled = [e for e in stored if e.get("ground_truth")]
            if labelled:
                from sentence_transformers import SentenceTransformer
                import numpy as np
                model = SentenceTransformer("all-MiniLM-L6-v2")
 
                query_vec = model.encode(f"{request.subject} {request.body}", convert_to_numpy=True)
                best, best_score = None, -1.0
                for e in labelled:
                    vec = model.encode(f"{e['subject']} {e['body']}", convert_to_numpy=True)
                    score = float(np.dot(query_vec, vec) /
                                  (np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-9))
                    if score > best_score:
                        best_score, best = score, e
 
                result = inference_service.classify_email(email)
                return EmailClassificationResponse(
                    predicted_topic=best["ground_truth"],
                    topic_scores=result["topic_scores"],
                    features=result["features"],
                    available_topics=result["available_topics"]
                )
            # fall through to topic mode if no labelled emails exist
 
        result = inference_service.classify_email(email)
        return EmailClassificationResponse(
            predicted_topic=result["predicted_topic"],
            topic_scores=result["topic_scores"],
            features=result["features"],
            available_topics=result["available_topics"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
 
@router.get("/topics")
async def topics():
    """Get available email topics"""
    inference_service = EmailTopicInferenceService()
    info = inference_service.get_pipeline_info()
    return {"topics": info["available_topics"]}
 
 
@router.get("/pipeline/info")
async def pipeline_info():
    inference_service = EmailTopicInferenceService()
    return inference_service.get_pipeline_info()
 

 
@router.post("/topics", response_model=TopicResponse, status_code=201)
async def create_topic(request: TopicCreateRequest):
    """Add a new topic to topic_keywords.json so it is available for classification."""
    try:
        topics = _load_topics()
        if request.name in topics:
            raise HTTPException(status_code=409, detail=f"Topic '{request.name}' already exists.")
        topics[request.name] = {
            "description": request.description,
            "keywords": request.keywords,
        }
        _save_topics(topics)
        return TopicResponse(name=request.name, description=request.description, keywords=request.keywords)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
 
@router.get("/topics/all", response_model=List[TopicResponse])
async def list_all_topics():
    """Return every topic with its description and keywords."""
    try:
        topics = _load_topics()
        return [TopicResponse(name=k, description=v["description"], keywords=v["keywords"])
                for k, v in topics.items()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
 
@router.delete("/topics/{name}")
async def delete_topic(name: str):
    """Remove a topic from topic_keywords.json."""
    try:
        topics = _load_topics()
        if name not in topics:
            raise HTTPException(status_code=404, detail=f"Topic '{name}' not found.")
        del topics[name]
        _save_topics(topics)
        return {"message": f"Topic '{name}' deleted."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
 
@router.post("/emails/store", response_model=StoredEmailResponse, status_code=201)
async def store_email(request: StoredEmailRequest):
    """
    Store an email for use in email-mode classification.
    Supply ground_truth (topic name) to make this email useful as a labelled reference.
    """
    try:
        if request.ground_truth:
            topics = _load_topics()
            if request.ground_truth not in topics:
                raise HTTPException(
                    status_code=422,
                    detail=f"ground_truth '{request.ground_truth}' is not a known topic. "
                           f"Known: {list(topics.keys())}"
                )
        emails = _load_emails()
        new_email = {
            "id": str(uuid.uuid4()),
            "subject": request.subject,
            "body": request.body,
            "ground_truth": request.ground_truth,
        }
        emails.append(new_email)
        _save_emails(emails)
        return StoredEmailResponse(**new_email)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
 
@router.get("/emails/stored", response_model=List[StoredEmailResponse])
async def list_stored_emails():
    """Return all stored emails."""
    try:
        return [StoredEmailResponse(**e) for e in _load_emails()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 

# TODO: LAB ASSIGNMENT - Part 2 of 2  
# Create a GET endpoint at "/features" that returns information about all feature generators
# available in the system.
#
# Requirements:
# 1. Create a GET endpoint at "/features"
# 2. Import FeatureGeneratorFactory from app.features.factory
# 3. Use FeatureGeneratorFactory.get_available_generators() to get generator info
# 4. Return a JSON response with the available generators and their feature names
# 5. Handle any exceptions with appropriate HTTP error responses
#
# Expected response format:
# {
#   "available_generators": [
#     {
#       "name": "spam",
#       "features": ["has_spam_words"]
#     },
#     ...
#   ]
# }
#
# Hint: Look at the existing endpoints above for patterns on error handling
# Hint: You may need to instantiate generators to get their feature names

