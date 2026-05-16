from typing import Dict, Any, Optional
from app.models.similarity_model import EmailClassifierModel
from app.features.factory import FeatureGeneratorFactory
from app.dataclasses import Email
import json
from pathlib import Path
 
EMAILS_FILE = Path("data") / "stored_emails.json"
 
 
class EmailTopicInferenceService:
    """Service that orchestrates email topic classification using feature similarity matching"""
 
    def __init__(self):
        self.model = EmailClassifierModel()
        self.feature_factory = FeatureGeneratorFactory()
 
    def classify_email(self, email: Email) -> Dict[str, Any]:
        """Classify an email into topics using generated features"""
 
        # Step 1: Generate features from email
        features = self.feature_factory.generate_all_features(email)
 
        # Step 2: Classify using features
        predicted_topic = self.model.predict(features)
        topic_scores = self.model.get_topic_scores(features)
 
        # Return comprehensive results
        return {
            "predicted_topic": predicted_topic,
            "topic_scores": topic_scores,
            "features": features,
            "available_topics": self.model.topics,
            "email": email
        }
 
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the inference pipeline"""
        return {
            "available_topics": self.model.topics,
            "topics_with_descriptions": self.model.get_all_topics_with_descriptions()
        }
 
 
    def classify_email_by_similarity(self, email: Email) -> Dict[str, Any]:
        """
        Classify an email by finding the most similar labelled stored email.
 
        Falls back to topic-mode classification if no labelled emails exist
        in the store, adding a 'fallback' key to the result so callers can
        tell which mode was actually used.
        """
        # Load stored emails from disk
        if EMAILS_FILE.exists():
            with open(EMAILS_FILE) as f:
                stored_emails = json.load(f)
        else:
            stored_emails = []
 
        # Ask the model to find the nearest neighbour
        predicted_topic = self.model.predict_from_stored_emails(
            query_subject=email.subject,
            query_body=email.body,
            stored_emails=stored_emails,
        )
 
        # Generate features and scores regardless (needed for the response)
        features = self.feature_factory.generate_all_features(email)
        topic_scores = self.model.get_topic_scores(features)
 
        if predicted_topic is not None:
            return {
                "predicted_topic": predicted_topic,
                "topic_scores": topic_scores,
                "features": features,
                "available_topics": self.model.topics,
                "email": email,
                "mode": "email",
            }
 
        # Fallback
        fallback_topic = self.model.predict(features)
        return {
            "predicted_topic": fallback_topic,
            "topic_scores": topic_scores,
            "features": features,
            "available_topics": self.model.topics,
            "email": email,
            "mode": "topic",
            "fallback": True,
            "fallback_reason": "No labelled emails in store; used topic mode",
        }
