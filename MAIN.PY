from flask import Flask, request, jsonify
import torch
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
import os
from google.cloud import storage

app = Flask(__name__)

# Google Cloud Storage bucket and model directory
BUCKET_NAME = os.getenv('MODEL_BUCKET', 'edu-feedback-model-bucket')
MODEL_DIR = "/tmp/model"

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)


# ✅ Function to download model files from GCS
def download_model_from_gcs():
    print("Downloading model files from GCS...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    
    # List of model files to download
    model_files = [
        "fine_tuned_edu_feedback_bert_model/model.safetensors",
        "fine_tuned_edu_feedback_bert_model/special_tokens_map.json",
        "fine_tuned_edu_feedback_bert_model/tokenizer_config.json",
        "fine_tuned_edu_feedback_bert_model/vocab.txt"
    ]
    
    for model_file in model_files:
        blob = bucket.blob(model_file)
        destination = os.path.join(MODEL_DIR, os.path.basename(model_file))
        blob.download_to_filename(destination)
        print(f"Downloaded {model_file} to {destination}")

    print("All model files downloaded successfully.")


# ✅ Function to load the model with 5 labels
def load_model():
    print("Loading model...")

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)

    # Set config with 5 labels
    config = BertConfig.from_pretrained(MODEL_DIR, num_labels=5)

    # Load model with 5 labels
    model = BertForSequenceClassification.from_pretrained(
        MODEL_DIR,
        config=config,
        ignore_mismatched_sizes=True  # Avoid shape mismatches
    )

    model.eval()
    print("Model loaded successfully.")
    return model, tokenizer


# ✅ Route for health check
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running'})


# ✅ Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Use request.get_json() to handle raw JSON payloads correctly
        data = request.get_json()

        if not data or 'student_answer' not in data or 'model_answer' not in data:
            return jsonify({'error': 'Invalid input format. Both student_answer and model_answer are required.'}), 400

        student_answer = data['student_answer']
        model_answer = data['model_answer']

        # Make prediction
        inputs = tokenizer(student_answer, model_answer, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)[0].tolist()

        labels = ["Excellent", "Good", "Average", "Below Average", "Poor"]
        result = {label: score for label, score in zip(labels, scores)}

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # ✅ 5-label mapping
    label_mapping = {
        0: "Label A",
        1: "Label B",
        2: "Label C",
        3: "Label D",
        4: "Label E"
    }

    predicted_label = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_label].item()

    result = {
        "label": label_mapping[predicted_label],
        "confidence": confidence
    }

    return jsonify(result)


# ✅ Initialize and run the app
if __name__ == '__main__':
    download_model_from_gcs()
    model, tokenizer = load_model()
    app.run(host='0.0.0.0', port=8080)
