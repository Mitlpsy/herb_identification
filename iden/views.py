# views.py (inside image_classification app)
from django.shortcuts import render
from .models import *
import tensorflow as tf
import numpy as np
import os
import cv2
from keras.models import load_model
from .class_mapping import class_mapping
import pickle


def preprocess_image(image_data):
    # Perform preprocessing operations on the image_data
    # e.g., resize, normalize, convert color space
    width = 160
    height = 120

    # Convert the image data to a NumPy array
    nparr = np.frombuffer(image_data.read(), np.uint8)

    # Read the image using cv2
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Resize the image to the model's input size
    preprocessed_image = cv2.resize(image, (height, width))

    # Normalize the pixel values to the range [0, 1]
    preprocessed_image = preprocessed_image.astype('float32') / 255.0

    # Add batch dimension to match the model's input shape
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)

    return preprocessed_image

def perform_prediction(model, preprocessed_image):
    # Perform prediction using the model on the preprocessed image
    # Return the predicted class

    # Assuming it's a binary classification, return the class with the highest probability
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions[0])

    return predicted_class


def predict_image(request):
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'model.pkl')
    if request.method == 'POST':
        # Retrieve the uploaded image from the request
        uploaded_file = request.FILES['image']

        # Preprocess the image if required
        preprocessed_image = preprocess_image(uploaded_file)

        # Load the trained model
        model = pickle.load(open(model_path, 'rb'))


        # Perform prediction using the model on the preprocessed image
        predicted_class = perform_prediction(model, preprocessed_image)
        class_namelocal, class_scientific_name, class_properties, class_use = get_class_info(predicted_class)

        # Create a folder with the predicted class name if it doesn't exist
        predict_folder = os.path.join('media', 'predict', class_scientific_name)
        os.makedirs(predict_folder, exist_ok=True)

        # Save the uploaded image in the folder with the predicted class name
        image_path = os.path.join(predict_folder, uploaded_file.name)
        with open(image_path, 'wb') as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)


        # Save the prediction result to the database
        prediction_obj = Prediction.objects.create(image=uploaded_file, result=predicted_class,
                                                   class_namelocal = class_namelocal, class_scientific_name = class_scientific_name, 
                                                   class_properties = class_properties, class_use = class_use)

        return render(request, 'prediction_result.html', {'prediction': prediction_obj})

    return render(request, 'upload_image.html')

# You can use the preprocess_image and perform_prediction functions from the previous answer here

def get_class_info(predicted_class):
    # Retrieve the class information from the class_mapping dictionary
    class_info = class_mapping.get(predicted_class)
    if class_info:
        class_namelocal = class_info["Local_name"]
        class_scientific_name = class_info["Scientific_name"]
        class_properties = class_info["Properties"]
        class_use = class_info["Use"]
    else:
        class_namelocal = "Unknown name"
        class_scientific_name = class_info["Unknown Scientific name"]
        class_properties = class_info["Unknown Properties"]
        class_use = class_info["Unknown how to Use"]
    return class_namelocal, class_scientific_name, class_properties, class_use
