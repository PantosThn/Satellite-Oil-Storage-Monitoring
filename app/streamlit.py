import streamlit as st
from PIL import Image
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import base64

st.title('Tank Volume Estimator')


page = st.sidebar.selectbox('Page Navigator', ['Predictor', 'Model analysis'])

st.sidebar.markdown("""___""")
st.sidebar.write("ECE dsml thesis")

def process(image, server_url: str):

    m = MultipartEncoder(
        fields={'file': ('filename', image, 'image/jpeg')}
        )

    r = requests.post(server_url,
                      data=m,
                      headers={'content-type': m.content_type,
                               'accept': 'application/json'},
                      timeout=8000)

    return r

if page == "Predictor":
    st.markdown("Select input satelite image.")
    upload_columns = st.columns([2, 1])

    #File upload
    file_upload = upload_columns[0].expander(label="Upload a satelite image")
    uploaded_file = file_upload.file_uploader("Choose an image file", type=['jpg'])
    input_image = st.file_uploader("insert image")  # image upload widget

    if input_image is not None:
        #st.write(type(uploaded_file))
        image = Image.open(uploaded_file)
        upload_columns[1].image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        #res = requests.post(f"http://127.0.0.1:8000/", files=image)
        segments = process(input_image, 'http://localhost:8000/prediction/')
        #label = predict(uploaded_file)
        st.write(segments.request.headers)
        st.write(segments.headers)
        st.write(segments.content)
        st.write(segments)

        enc_img = eval(segments.content['image_encoded'])
        img = base64.b64decode(enc_img)
        st.write(img)
