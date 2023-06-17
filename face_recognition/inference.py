from typing import Union

import numpy as np
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from facenet_pytorch.models.mtcnn import MTCNN
from PIL.Image import Image
from sklearn.metrics.pairwise import cosine_similarity

from face_recognition.messages import Message


class FaceRecognizer:
    def __init__(
        self,
        model: InceptionResnetV1,
        mtcnn: MTCNN,
        threshold: float,
        persons: np.ndarray,
        vectors: np.ndarray,
    ) -> None:
        self.model = model
        self.mtcnn = mtcnn
        self.threshold = threshold
        self.persons = persons
        self.vectors = vectors

    def recognize(self, img: Image):
        preds = inference_img(img, self.mtcnn, self.model)
        if preds is None:
            return Message.FACE_NOT_FOUND
        sims = cosine_similarity(preds, self.vectors)
        # 対象とのコサイン類似度が高いインデックスを取得
        high_sim_index = np.argmax(sims[0])
        sim_score = sims[0, high_sim_index]
        # 最も似ている顔画像ベクトルのインデックスがしきい値以下の場合は
        # 登録された顔がないメッセージを返す
        if sim_score < self.threshold:
            return Message.NOT_REGISTERED
        # 最も似ている人物の名前を返す
        return self.persons[high_sim_index]


def inference_img(
    img: Image, mtcnn: MTCNN, model: InceptionResnetV1
) -> Union[np.ndarray, None]:
    img_cropped = mtcnn(img)
    if img_cropped is None:
        return None
    preds = model(img_cropped.unsqueeze(0)).detach().cpu().numpy()
    return preds
