import os
from flask import Flask, render_template, request
from PIL import Image
import torch
from torchvision import transforms
from transformers import SwinConfig, SwinForImageClassification

# ------------------ Flask Setup ------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ------------------ Device ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Class Names ------------------
class_names = ["Karacadag", "Arborio", "Basmati", "Ipsala", "Jasmine"]

# ------------------ Load Model ------------------
def load_model():
    # Load base Swin configuration
    config = SwinConfig.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    config.num_labels = 5

    # Initialize model architecture
    model = SwinForImageClassification(config)

    # Load your trained weights
    state_dict = torch.load("swin_rice_model.pth", map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model

model = load_model()

# ------------------ Preprocessing ------------------
def preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image.to(device)

# ------------------ Routes ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", prediction="No file uploaded")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", prediction="No file selected")

        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            try:
                image_tensor = preprocess_image(file_path)

                with torch.no_grad():
                    outputs = model(pixel_values=image_tensor)
                    logits = outputs.logits
                    pred_idx = torch.argmax(logits, dim=1).item()
                    prediction = class_names[pred_idx]

            except Exception as e:
                prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

# ------------------ Run ------------------
if __name__ == "__main__":
    app.run(debug=True)
