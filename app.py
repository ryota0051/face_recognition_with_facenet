import json

import numpy as np
import streamlit as st
from facenet_pytorch import MTCNN, InceptionResnetV1

from face_recognition.data_utils import open_and_convert_img2RGB
from face_recognition.inference import FaceRecognizer

with open("./config.json", "r") as f:
    config = json.load(f)

model = InceptionResnetV1(pretrained="vggface2").eval()
mtcnn = MTCNN()

persons_and_face_vectors = np.load("./data/vectors.npy", allow_pickle=True)[()]

face_recognizer = FaceRecognizer(
    model,
    mtcnn,
    threshold=config["threshold"],
    persons=persons_and_face_vectors["persons"],
    vectors=persons_and_face_vectors["face_vector"],
)


def main():
    st.title("Face Recognition APP")
    uploaded_file = st.file_uploader("Choose a image file", type="jpg")
    if uploaded_file is not None:
        image = open_and_convert_img2RGB(uploaded_file)
        result = face_recognizer.recognize(image)
        text = f'<p style="font-family:sans-serif; font-size: 42px;">{result}</p>'
        st.image(np.array(image), caption=result)
        st.markdown(text, unsafe_allow_html=True)


main()
