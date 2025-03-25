
### 🛠️ **Edu Feedback AI – Automated Grading & Feedback System**

![GitHub Release](https://img.shields.io/github/release/Subhrasameerdash/gdg-codes.svg)				![GitHub last commit](https://img.shields.io/github/last-commit/Subhrasameerdash/gdg-codes)				![GitHub repo size](https://img.shields.io/github/repo-size/Subhrasameerdash/gdg-codes) 				[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)




---

### 🚀 **Project Overview**

Edu Feedback AI is an AI-powered automated grading and feedback system designed for **Class 11 & 12 students' assignments**. The system streamlines the evaluation process by leveraging **Google Cloud services**, **BERT-based models**, and **custom NLP pipelines** for accurate and consistent feedback.

---

### 🔥 **Key Features**

✅ **Automated Grading:**  
- Uses **fine-tuned BERT models** to automatically grade student submissions.  
- Supports **short answers, essays, and code submissions**.  

✅ **Personalized Feedback:**  
- Provides **detailed, constructive feedback** for each question.  
- Includes improvement suggestions and learning tips.  

✅ **Google Cloud Integration:**  
- Uses **Google Cloud Functions** for serverless execution.  
- Stores large model files in **Cloud Storage**.  
- Releases provide external downloads for large files.

✅ **GitHub Release for Large Models:**  
- Due to GitHub's size limitations, large models are uploaded as **Releases** and can be downloaded using the included Python script.  

✅ **Efficient Data Pipeline:**  
- Includes **PDF to text conversion**, **text cleaning**, and **model fine-tuning** pipelines.  

---

### 📁 **Repository Structure**

```
/edu-feedback-api               → Backend API with Flask & Google Cloud Functions.  
/evaluate_model.py              → Model evaluation script.  
/download_models.py             → Script to download large models from GitHub Releases.  
/grading_reports                → Grading results and logs.  
/logs                           → Logs from model runs.  
/model_1, model_2, model_3      → Fine-tuned BERT models.  
/training_data                  → Training dataset for fine-tuning.  
/teacher_assignments            → Sample student submissions for testing.  
/subjectwise_dataset.ipynb      → Dataset creation notebook.  
/training_dataset.json          → Sample dataset used for model training.  
```

---

### 🔥 **Installation and Usage**

#### ✅ 1. Clone the repository:
```bash
git clone https://github.com/Subhrasameerdash/gdg-codes.git
cd gdg-codes
```

#### ✅ 2. Install dependencies:
```bash
pip install -r requirements.txt
```

#### ✅ 3. Download model files from Releases:
```bash
python download_models.py
```
This will download the **model_1, model_2, and model_3** files and place them in the appropriate directories.

#### ✅ 4. Run the API locally:
```bash
cd edu-feedback-api
python main.py
```
API will be available at:  
```bash
http://localhost:8080
```

#### ✅ 5. Test API with Postman or CURL:
```bash
curl -X POST "http://localhost:8080/predict" -H "Content-Type: application/json" -d @test_payload.json
```

---

### 🔥 **Model Releases**

The following large models are stored as **GitHub Releases** due to their size:  

| Model            | Download URL                                    | Size     |
|-----------------|-------------------------------------------------|----------|
| `model_1`       | [model_1.safetensors](https://github.com/Subhrasameerdash/gdg-codes/releases/download/v1.0/model_1.safetensors) | 417.96 MB |
| `model_2`       | [model_2.safetensors](https://github.com/Subhrasameerdash/gdg-codes/releases/download/v1.0/model_2.safetensors) | 417.68 MB |
| `model_3`       | [model_3.safetensors](https://github.com/Subhrasameerdash/gdg-codes/releases/download/v1.0/model_3.safetensors) | 417.45 MB |

✅ **Download them using the `download_models.py` script.**

---

### 🔥 **Technologies Used**

✅ **Languages:** Python, Flask, Bash  
✅ **ML Libraries:** TensorFlow, PyTorch, Transformers  
✅ **Cloud Services:** Google Cloud Functions, Cloud Storage  
✅ **Version Control:** Git, GitHub Releases  
✅ **Deployment:** Google Cloud Functions with API  

---

### 🚀 **How to Contribute**

1. **Fork** the repository.  
2. Create a **feature branch** (`git checkout -b feature-name`).  
3. Commit your changes (`git commit -m "Add feature"`).  
4. Push to the branch (`git push origin feature-name`).  
5. Create a **Pull Request**.  

---

### 🛡️ **License**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
  
This project is licensed under the **MIT License**.  
See the [LICENSE](./LICENSE) file for details.

---

### 💬 **Contact**

📧 **Author:** [Subhrasameer Dash](https://github.com/Subhrasameerdash)  
💬 **GitHub:** [Edu Feedback AI](https://github.com/Subhrasameerdash/gdg-codes)  

🔥 If you have any issues, feel free to **open an issue** or reach out.

---

✅ **This `README.md`** provides a detailed overview of your project, including setup, usage, and references to your GitHub Releases.🚀