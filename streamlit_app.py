import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

st.title('Gelada Classifier')

st.info("This is a Gelada classifier Machine Learning application!")

image_uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Importing the model
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("model/gelada_classifier.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Transform image for resnet50 specifications
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class_names = ['northern', 'southern']


# Applying model to the uploaded image
if image_uploaded is not None:

    # Convert the uploaded binary image to an image object, and display it
    cur_image = Image.open(image_uploaded).convert("RGB")
    st.image(cur_image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for the resnet50 model
    input_tensor = transform(cur_image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        pred_index = torch.argmax(probabilities, dim=1).item()
        predicted_label = class_names[pred_index]
        confidence = probabilities[0][pred_index].item()
    
    st.info(f"Prediction: {predicted_label} Gelada")