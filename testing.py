import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import sys

# -----------------------------
# Define CustomCNN (must match training)
# -----------------------------
class CustomCNN(nn.Module):
    """Custom Shallow CNN for binary classification"""
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224 -> 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112 -> 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56 -> 28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28 -> 14

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))  # -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = r"c:\Users\Ammad\Documents\Projects\Personal\Brain2\models\best_model.pth"
CLASS_NAMES = ["MRI", "BreastHisto"]  # must match training label order

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------

checkpoint = torch.load(MODEL_PATH, map_location=device)


model = CustomCNN(num_classes=len(CLASS_NAMES), dropout_rate=0.5)


model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print(f"✅ Model loaded from {MODEL_PATH} (epoch-trained) on {device}")


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # your CNN expects 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # same as training
])

def predict_image(image_path):
    """Load image, preprocess, predict class"""
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return
    
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probs, 1)
    
    print(f"🔮 Prediction: {CLASS_NAMES[pred_class.item()]} (confidence: {confidence.item():.4f})")

if __name__ == "__main__":
    print("🧠 CustomCNN Prediction Tool")
    print("Type 'exit' to quit.")
    while True:
        img_path = input("\nEnter image path: ").strip()
        if img_path.lower() == "exit":
            print("👋 Exiting...")
            break
        predict_image(img_path)
