import pandas as pd
import streamlit as st
import numpy as np
import tensorflow as tf
import base64
from PIL import Image , ImageOps
from tensorflow.compat.v1.keras import backend as K



@st.cache(allow_output_mutation=True)
def load_model_catsdogs():

	model = tf.keras.models.load_model('dogs_and_cats_VGGclassifier.h5')
	#model._make_predict_function()
	model.summary()  # included to make it visible when model is reloaded
	session = K.get_session()
	return model,session

@st.cache(allow_output_mutation=True)
def gift():

		file_ = open("catdog.gif", "rb")
		contents = file_.read()
		data_url = base64.b64encode(contents).decode("utf-8")
		file_.close()
		return data_url


def image_upload(img):
		
		images = Image.open(img)
		st.image(images, caption='Uploaded Image.', use_column_width=True)
		st.write("")
		st.write("Classifying...")
		return images
		
def Model(img, model):
		images = img
		data = np.ndarray(shape=(1,256, 256, 3), dtype=np.float32)
		im = ImageOps.fit(images, (256,256))
		image_array = np.asarray(im)
		data[0] = image_array

		result = model.predict(data)

		if result[0][0] == 1:
			prediction = 'Dog'
			var = ':dog:'	
		else:
			prediction = 'Cat'
			var = ':cat:'
		st.title(var)
		st.write('Predicted category is:', prediction)


  
def main():
	st.title("Image Classification with Convolution Neural Network")

	analysis = st.sidebar.selectbox("Index", ["Problem", "Cats v/s Dogs"])
#=======================================================================================================================================
# EXPLAINING THE CNN	
#=======================================================================================================================================	
	if analysis == "Problem":

		st.header("Cats vs Dogs - A Binary classification problem")
		st.text("A cat/dog Image classification model to classify it as cat or dog")
		st.image('catdog1.jpeg', width= 700, use_column_width=True)
		
		st.subheader("CNN Model")

		data_url = gift()
		st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',unsafe_allow_html=True)

#=======================================================================================================================================	
# CATS & DOGS MODEL
#=======================================================================================================================================			
	elif analysis == "Cats v/s Dogs":	
		st.header("Classification Model")
		st.subheader("Problem Type: Dogs V/S Cats Classifier")
		model, session= load_model_catsdogs()		
		K.set_session(session)
		uploaded_file = st.file_uploader("Choose an image...", type="jpg")
		if uploaded_file is not None:
			test_image= image_upload(uploaded_file)
			Model(test_image,model)

main()