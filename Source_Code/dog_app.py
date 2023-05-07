import streamlit as st
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from streamlit_option_menu import option_menu

#Custom CSS *************************************************
st.markdown("""
        <style>
            .css-uf99v8{
                background: linear-gradient(191deg, rgba(0,75,102,1) 9%, rgba(131,0,130,1) 31%, rgba(77,0,187,1) 67%, rgba(5,179,155,1) 100%);
            }
            .css-6qob1r{
                background-color: #0b0c20
            }
            .st-bb {
                opacity: 1;
                background-color: #0b0c20;
                border-radius: 21px;
            }
            .css-z8f339{
                padding: 3rem;
                background-color: #0b0c20;
                border-radius: 21px;
            }
        </style>

""", unsafe_allow_html=True)



#************************************************************


# Function to load Model ****************
def load_model(model_path):
    print(f"Loading a model from: {model_path}")
    model_h5 = tf.keras.models.load_model(model_path, custom_objects={"KerasLayer": hub.KerasLayer})
    return model_h5

# Loading pre-trained dog breed classification model here
model = load_model('Models/dog_breed_model.h5')

# ******Preprocessing Image*********
# Defining image size
IMG_SIZE = 224

# Creating function to preprocess images
def process_images(image_path):
    # Read image file
    image = tf.io.read_file(image_path)

    # Turn jpg image to numerical Tensors with 3 colors channels (RGB)
    image = tf.image.decode_jpeg(image, channels=3)

    # Convert the color channel values from 0-255 to 0-1 (Normalization)
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resizing the image to (224, 224)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])

    return image

# Creating Batch Sizes********************
# Defining Batch Size (32)
BATCH_SIZE = 32

# Creating function to turn data into batches
def create_data_batches(X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    """
    Creates a batch of data out of image(X) and label (y) pairs
    Shuffles the data if it's training data but doesn't shuffle if it's validation data.
    Also accepts test data as input (no labels).
    """
    # If data is a test dataset, we probably don't have labels
    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))  # only filepaths (no labels)
        data_batch = data.map(process_images).batch(BATCH_SIZE)
        return data_batch
    else:
        pass

# Decoding Prediction********
labels_csv = pd.read_csv('Datasets/labels.csv')
labels = labels_csv.breed
labels = np.array(labels)
unique_breeds = np.unique(labels)

def decode_prediction(prediction_probabilities):
    return unique_breeds[np.argmax(prediction_probabilities)]

# Making Prediction***************
def classify_dog_breed(image):
    # Save the PIL image as a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_filename = temp_file.name
        image.save(temp_filename)

    # Preprocess the image if required (resize, normalize, etc.)
    image_path = temp_filename
    image_data = create_data_batches([image_path], test_data=True)

    # Using the pre-trained model to classify the dog breed
    global prediction
    prediction = model.predict(image_data)
    #breed = decode_prediction(prediction)
    top_5_indices = np.argsort(prediction[0])[::-1][:5]  # Get indices of top 5 predictions
    top_5_breeds = unique_breeds[top_5_indices]
    top_5_scores = prediction[0][top_5_indices]

    # Delete the temporary file
    os.remove(temp_filename)

    return top_5_breeds, top_5_scores

#Main Function***********************
def main():
    st.markdown("""
    <h1 style='text-align: center;
    background-color:#0b0c20;
    margin-bottom:35px;
    border-radius:21px'>Dog Breed Classifier</h1>
    """, unsafe_allow_html=True)
    st.info('Upload an image or capture an image using the camera', icon="ℹ️")

    # Sidebar selection
    option = st.sidebar.radio("Select Option", ("Upload Image", "Capture Image"))

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)
            if st.button("Classify"):
                top_5_breeds, top_5_scores = classify_dog_breed(image)
                st.write("The breed of the dog above is: ", decode_prediction(prediction))
                st.write("Accuracy of the result is: {:.2f}%".format(np.max(prediction)*100))

                # Create a DataFrame with the top 5 predictions and scores
                dataframe = pd.DataFrame({"Breed": top_5_breeds, "Accuracy": top_5_scores * 100})
                # Plotting the bar graph
                st.header("Other Possibilities based on Accuracy Level")
                st.bar_chart(data=dataframe, x='Breed', y='Accuracy')

    elif option == "Capture Image":
        st.write("Please grant access to your camera")
        camera_input = st.camera_input("Capture the photo of a dog")
        if camera_input is not None:
            image2 = Image.open(camera_input)
            st.image(image2, caption="Uploaded Image", width=300)
        if st.button("Classify"):
            top_5_breeds, top_5_scores = classify_dog_breed(image2)
            # Create a DataFrame with the top 5 predictions and scores
            dataframe2 = pd.DataFrame({"Breed": top_5_breeds, "Accuracy": top_5_scores * 100})
            st.write("The breed of the dog above is: ", top_5_breeds[0])
            st.write("Accuracy of the result is: {:.2f}%".format(top_5_scores[0]*100))
            # Plotting the bar graph
            st.header("Other Possibilities based on Accuracy Level")
            st.bar_chart(data=dataframe2, x='Breed', y='Accuracy')


    st.sidebar.write("---")
    st.sidebar.write("Made with 🧠 by Abhinash")

if __name__ == "__main__":
    main()
