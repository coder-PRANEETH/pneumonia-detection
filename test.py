

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

                                         
                                         
                                         
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
DATASET_DIR = "dataset"                                            

                                         
                                         
                                         
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

                                         
                                         
                                         
train_dataset = datasets.ImageFolder(
    root=f"{DATASET_DIR}/TRAIN",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root=f"{DATASET_DIR}/TEST",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Classes:", train_dataset.classes)
                                         

                                         
                                         
                                         
model = models.densenet169(
    weights=models.DenseNet169_Weights.IMAGENET1K_V1
)

                                         
model.classifier = nn.Identity()

                                         
for param in model.parameters():
    param.requires_grad = False

model.to(DEVICE)
model.eval()

                                         
                                         
                                         
def extract_features(model, loader):
    features = []
    labels = []

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            feats = model(imgs)                                            
            features.append(feats.cpu())
            labels.append(lbls)

    X = torch.cat(features).numpy()
    y = torch.cat(labels).numpy()
    return X, y

print("Extracting TRAIN features...")
X_train, y_train = extract_features(model, train_loader)

print("Extracting TEST features...")
X_test, y_test = extract_features(model, test_loader)

print("Feature shape:", X_train.shape)                                           

                                         
                                         
                                         
svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=3.5,
        gamma=0.9e-05,
        probability=True
    ))
])

svm.fit(X_train, y_train)

                                         
                                         
                                         
y_pred = svm.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", acc)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=train_dataset.classes
))

                                         
                                         
                                         
def predict_image(image_path):
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = model(img).cpu().numpy()

    pred = svm.predict(feat)[0]
    prob = svm.predict_proba(feat)[0]

    class_name = train_dataset.classes[pred]
    return class_name, prob

                                         
                                 
