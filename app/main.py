import os
import json
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st

working_dir=os.path.dirname(os.path.abspath(__file__))
model_path =f"{working_dir}/trained_model/Cnn_plant_disease1.pkl"

#Define the model class
class PlantDisease (nn.Module):
    def __init__ (self,inputsize,batch_size,out_size,shape):
        super().__init__()
        self.input_size=inputsize
        self.batchsize=batch_size
        self.out_size=out_size
        self.x_in=int((shape/4)**2 * 64)
        
        self.layer1=nn.Sequential(nn.Conv2d(self.input_size,32,3,padding=1),
                             nn.ReLU(),
                             nn.MaxPool2d(2),
                             nn.Conv2d(32,64,3,padding=1),
                             nn.MaxPool2d(2),
                             nn.ReLU())
        self.linear=nn.Sequential(nn.Linear(self.x_in,256),
                                  nn.ReLU(),
                                  nn.Linear(256,self.out_size))

    def forward(self,x):
        self.batchsize=x.shape[0]
        print(x.shape)
        x=self.layer1(x)
        print(x.shape)
        x=self.linear (x.contiguous().view(self.batchsize,-1))
        return (x)
    
#load the model
model=PlantDisease(3,1,38,224)
model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')
                                 ))

#Loading class names
class_indices=json.load(open(f"{working_dir}/trained_model/class_idices.json"))


#Function to Load and Preprocess the Image using Pillow

def load_and_preprocess_image(image_path,target_size=(224,224)):
    #load image
    img=Image.open(image_path)

    #resize the Image
    img=img.resize(target_size)

    img_array=np.array(img)

    # print(img_array.shape)

    #Convert image in to tensor
    img_tensor = torch.from_numpy(img_array)
    
    #scale the image from 0 to 1
    img_tensor=img_tensor.float()/255.0

    #Add batch dimension
    img_tensor=img_tensor.unsqueeze(0)

    img_tensor=img_tensor.permute(0, 3, 1, 2)
    # print(img_tensor.shape)

    return img_tensor

# Function to predict the class of the image

def predict_image_class(model,image_path,class_indices):

    pre_processed_image=load_and_preprocess_image(image_path)
    prediction=model(pre_processed_image)
    _, predicted=torch.max(prediction,1)
    predicted_class_name=class_indices[str(predicted.item())]

    return predicted_class_name

#Streamlit app
st.title('🪴 Plant Disease Classifier')

uploaded_image=st.file_uploader("Upload an image...",type=['jpg','jpeg','png'])

if uploaded_image is not None:

    image=Image.open(uploaded_image)
    col1,col2=st.columns(2)

    with col1:
        resized_img = image.resize((150,150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            #Preprocess the uploaded image and predict the class
            prediction=predict_image_class(model, uploaded_image,class_indices)
            st.success(f'Prediction: {str(prediction)}')