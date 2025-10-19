"""
Streamlit Web App for MNIST Classifier
BONUS TASK: Deploy Your Model
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms

# Define the same CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Streamlit App
st.title("🔢 MNIST Digit Classifier")
st.write("Upload a handwritten digit image (28x28 pixels)")

uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption='Uploaded Image', width=200)
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    img_tensor = transform(image).unsqueeze(0)
    
    # Load model (you need to train and save it first!)
    # model = CNN()
    # model.load_state_dict(torch.load('mnist_model.pth'))
    # model.eval()
    
    # For demo, we'll use random prediction
    st.write("### Prediction: Coming soon!")
    st.write("(Train and save your model first, then uncomment the code)")

st.write("---")
st.write("Made with ❤️ for AI Tools Assignment")