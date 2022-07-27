import streamlit as st
from PIL import Image
import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import base64
import json

st.set_page_config(layout="wide")
st.title('Tank Volume Estimator')

page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

st.sidebar.title("ECE DSML Thesis")
st.sidebar.image("/home/thanos/dev/thesis_refactor/app/images/NTUA-logo.png", width=200)
st.sidebar.markdown("""___""")
page = st.sidebar.selectbox('Page Navigator', ['Estimate Volume'])

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

def n_barrels_calculator(height, diameter):
    radius = diameter/2
    return (radius * radius * height * 3.14 * 6.28)


if page == "Estimate Volume":
    st.markdown("Select input satelite image.")
    upload_columns = st.columns([0.3, 0.2, 0.4])

    #File upload
    file_upload = upload_columns[0].expander(label="Upload a satelite image")
    uploaded_file = file_upload.file_uploader("Choose an image file", type=['jpg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        upload_columns[1].image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.header("Classifying...")
        st.write("")
        st.write("")
        st.write("")
        segments = process(uploaded_file, 'http://localhost:8000/prediction/')
        my_json = segments.content.decode('utf8').replace("'", '"')
        data = json.loads(my_json)
        img = base64.b64decode(data['image_encoded'])
        col1, col2, col3 = st.columns([0.5, 0.1, 1.5])

        results = data['results']
        n_fht = len([i for i in results[0]])

        diameter = col1.number_input('Set the average floating head tank diameter in meters', value=8., format="%.2f")
        height = col1.number_input('Set the average floating head tank height in meters', value=25., format="%.2f")
        n_barrels = n_barrels_calculator(height, diameter)

        total_barrels = sum([float(i["volumes"])*n_barrels for i in results[0]])

        col1.metric('Number of Floating Head Tanks', n_fht)
        col1.metric('Number of Total Barrels', '{:,d}'.format(int(total_barrels)))
        col3.image(img, caption="Floating Head Tanks detected with the YOLOv3 Model and Localized with Bounding Boxes", use_column_width="auto")

        st.header("JSON Response")
        st.write(data['results'])
