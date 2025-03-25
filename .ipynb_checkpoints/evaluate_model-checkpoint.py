import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from transformers import BertTokenizer, BertForSequenceClassification

# === Paths to models ===
MODEL_PATHS = [
    "model_1",
    "model_2",
    "model_3"
]

# === Load the test dataset ===
TEST_DATA_PATH = "test_dataset.json"

# ✅ Load the test data
with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)

# === Function to evaluate a model ===
def evaluate_model(model_dir, test_data):
    print(f"\n🔥 Loading model: {model_dir}...")

    # ✅ Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # === Initialize results ===
    true_scores = []
    pred_scores = []
    true_grades = []
    pred_grades = []

    # ✅ Grade-to-score mapping
    grade_to_score = {
        "A": 5, "B": 4, "C": 3, "D": 2, "F": 1
    }

    # ✅ Inference loop
    for sample in test_data:
        student_answer = sample["input"]["student_answer"]
        model_answer = sample["input"]["model_answer"]

        # Tokenize input
        inputs = tokenizer(
            student_answer + " [SEP] " + model_answer,
            return_tensors="pt",
            max_length=512,
            padding="max_length",
            truncation=True
        ).to(device)

        # ✅ Model inference
        with torch.no_grad():
            outputs = model(**inputs)
            pred_score = torch.argmax(outputs.logits, dim=1).item()

        # ✅ Map scores to grades
        pred_grade = (
            "A" if pred_score >= 90 else
            "B" if pred_score >= 75 else
            "C" if pred_score >= 60 else
            "D" if pred_score >= 45 else
            "F"
        )

        # ✅ Append results
        true_scores.append(sample["output"]["score"])
        pred_scores.append(pred_score)
        true_grades.append(sample["output"]["grade"])
        pred_grades.append(pred_grade)

    # ✅ Convert grades to numerical scores
    true_numeric = [grade_to_score.get(g, 0) for g in true_grades]
    pred_numeric = [grade_to_score.get(g, 0) for g in pred_grades]

    # ✅ Calculate classification metrics
    accuracy = accuracy_score(true_grades, pred_grades)
    precision = precision_score(true_grades, pred_grades, average="weighted", zero_division=0)
    recall = recall_score(true_grades, pred_grades, average="weighted", zero_division=0)
    f1 = f1_score(true_grades, pred_grades, average="weighted", zero_division=0)

    # ✅ Calculate regression metrics
    mae = mean_absolute_error(true_scores, pred_scores)
    mse = mean_squared_error(true_scores, pred_scores)

    # ✅ Display results
    print("\n✅ Model Evaluation Results:")
    print(f"🔥 Classification Metrics for {model_dir}:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

    print(f"\n🔥 Regression Metrics for {model_dir}:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")

    # ✅ Return the results
    return {
        "model": model_dir,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mae": mae,
        "mse": mse
    }


# ✅ Iterate over all models and evaluate them
all_results = []
for model_path in MODEL_PATHS:
    result = evaluate_model(model_path, test_data)
    all_results.append(result)

# ✅ Display comparison results
print("\n🔥 🔥 🔥 Comparison of All Models 🔥 🔥 🔥\n")
for result in all_results:
    print(f"\n✅ Model: {result['model']}")
    print(f"Accuracy: {result['accuracy']:.2f}")
    print(f"Precision: {result['precision']:.2f}")
    print(f"Recall: {result['recall']:.2f}")
    print(f"F1-Score: {result['f1_score']:.2f}")
    print(f"Mean Absolute Error (MAE): {result['mae']:.2f}")
    print(f"Mean Squared Error (MSE): {result['mse']:.2f}")
