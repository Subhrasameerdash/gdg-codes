import os
import urllib.request

# ✅ URLs from GitHub release
model_urls = {
    "model_1": "https://github.com/Subhrasameerdash/gdg-codes/releases/download/v1.0/model_1.safetensors",
    "model_2": "https://github.com/Subhrasameerdash/gdg-codes/releases/download/v1.0/model_2.safetensors",
    "model_3": "https://github.com/Subhrasameerdash/gdg-codes/releases/download/v1.0/model_3.safetensors"
}

# ✅ Directory to save the models
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

# ✅ Download and save the models
for model_name, url in model_urls.items():
    print(f"Downloading {model_name}...")
    output_path = os.path.join(output_dir, f"{model_name}.safetensors")
    urllib.request.urlretrieve(url, output_path)
    print(f"{model_name} saved to {output_path}")
