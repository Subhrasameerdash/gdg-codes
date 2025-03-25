from flask import Flask, request, jsonify
from google.cloud import storage
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
from safetensors.torch import load_file
from difflib import SequenceMatcher
from Levenshtein import ratio
from flask_cors import CORS

# ✅ Flask app initialization
app = Flask(__name__)
CORS(app)

# ✅ GCS Config
BUCKET_NAME = "edu-feedback-model-bucket"
MODEL_PATH = "edu_feedback_bert_model_tuned"
LOCAL_MODEL_DIR = "/tmp/fine_tuned_edu_feedback_bert_model"

# ✅ Function to download the model from GCS
def download_model_from_gcs():
    """Download the model from Google Cloud Storage into the local file system."""
    if os.path.exists(LOCAL_MODEL_DIR) and len(os.listdir(LOCAL_MODEL_DIR)) > 0:
        print("✅ Model already exists locally. Skipping download.")
        return

    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

    # GCS storage client
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    # ✅ Define GCS blobs
    model_blob = bucket.blob(f"{MODEL_PATH}/model.safetensors")
    config_blob = bucket.blob(f"{MODEL_PATH}/config.json")
    tokenizer_blob = bucket.blob(f"{MODEL_PATH}/tokenizer_config.json")
    special_tokens_blob = bucket.blob(f"{MODEL_PATH}/special_tokens_map.json")
    vocab_blob = bucket.blob(f"{MODEL_PATH}/vocab.txt")

    # ✅ Download to local directory
    model_blob.download_to_filename(os.path.join(LOCAL_MODEL_DIR, "model.safetensors"))
    config_blob.download_to_filename(os.path.join(LOCAL_MODEL_DIR, "config.json"))
    tokenizer_blob.download_to_filename(os.path.join(LOCAL_MODEL_DIR, "tokenizer_config.json"))
    special_tokens_blob.download_to_filename(os.path.join(LOCAL_MODEL_DIR, "special_tokens_map.json"))
    vocab_blob.download_to_filename(os.path.join(LOCAL_MODEL_DIR, "vocab.txt"))

    print("✅ Model downloaded successfully!")
# ✅ Function to load the model
def load_model():
    """Load the model and tokenizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR)

    # Load safetensors weights
    safetensor_file = os.path.join(LOCAL_MODEL_DIR, "model.safetensors")

    if os.path.exists(safetensor_file):
        print("✅ Loading safetensors weights...")
        weights = load_file(safetensor_file)

        # Map weights properly
        missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
        
        if missing_keys or unexpected_keys:
            print(f"⚠️ Missing keys: {missing_keys}")
            print(f"⚠️ Unexpected keys: {unexpected_keys}")
        else:
            print("✅ Weights loaded successfully!")
    else:
        print("⚠️ Safetensors file not found, using default model weights.")
    
    model.to(device)
    return model, tokenizer, device
# ✅ Initialize model and tokenizer at server startup
model, tokenizer, device = load_model()

# ✅ Function to grade answers
def grade(student_answer, model_answer):
    """
    Grades the student's answer based on similarity to the model answer.
    """
    # Use Levenshtein ratio for better similarity measurement
    similarity = ratio(student_answer, model_answer)  # Value between 0 and 1

    # Scale the score to 0-100
    score = int(similarity * 100)

    # Map score to grade
    if score >= 85:
        grade = "A"
        feedback = "Excellent work! Keep it up!"
    elif score >= 70:
        grade = "B"
        feedback = "Good job, but you can add more details."
    elif score >= 55:
        grade = "C"
        feedback = "Fair effort, but needs more accuracy."
    elif score >= 40:
        grade = "D"
        feedback = "Below average, focus on key concepts."
    else:
        grade = "F"
        feedback = "Poor understanding, consider revising thoroughly."

    return grade, score, feedback
# ✅ Cloud Functions entry point
@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for grading student answers.
    """
    data = request.get_json()
    student_answer = data.get("student_answer", "")
    model_answer = data.get("model_answer", "")

    # Validate input
    if not student_answer or not model_answer:
        return jsonify({"error": "Invalid input"}), 400

    # Grade the answers
    grade_letter, score, feedback = grade(student_answer, model_answer)

    response = {
        "student_answer": student_answer,
        "model_answer": model_answer,
        "score": score,
        "grade": grade_letter,
        "feedback": feedback
    }

    return jsonify(response)

if __name__ == "__main__":
    # Use PORT environment variable for Cloud Run compatibility
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)

# ✅ Add this health check endpoint to main.py
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for Cloud Run"""
    return jsonify({"status": "healthy"}), 200
