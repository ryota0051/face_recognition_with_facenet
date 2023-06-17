from glob import glob

import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from face_recognition.data_utils import open_person_img
from face_recognition.inference import inference_img

PESON_AND_VECTOR_DST = "./data/vectors.npy"


def evaluate():
    """trainに対してベクトル化としきい値を計算。しきい値計算方法は下記の通り
    1. trainデータを1枚ずつ読み込んで顔画像をベクトル化
    2. コサイン類似度行列計算
    3. 類似度行列からf1_scoreをしきい値0.1刻みで計算して、最もf1_scoreが高いしきい値を出力
    """
    # モデルのロード
    model = InceptionResnetV1(pretrained="vggface2").eval()
    mtcnn = MTCNN()

    # tran用の画像ファイルパス取得
    train_img_path_list = list(glob("./data/train/**/*.jpg", recursive=True))
    person_list = []
    face_vector_list = []
    # 画像ごとに予測ベクトル化
    for img_path in tqdm(train_img_path_list):
        img, person_name = open_person_img(img_path)
        preds = inference_img(img, mtcnn, model)
        person_list.append(person_name)
        for p in preds:
            face_vector_list.append(p)
    assert len(person_list) == len(face_vector_list)
    save_dict = {
        "persons": np.array(person_list),
        "face_vector": np.array(face_vector_list),
    }
    np.save(PESON_AND_VECTOR_DST, save_dict)

    # コサイン類似度を計算
    sims = cosine_similarity(face_vector_list, face_vector_list)

    scores = {}
    # しきい値ごとにf1_scoreを計算する。
    for t in range(1, 100, 1):
        th = t / 100
        match_pred = []
        match_true = []
        for i in range(len(face_vector_list)):
            for j in range(len(face_vector_list)):
                if i == j:
                    continue
                # 予測人物のラベル: 0 => 別人, 1 => 同一人物
                if sims[i, j] < th:
                    match_pred.append(0)
                else:
                    match_pred.append(1)
                # 実際の人物ラベル: 0 => 別人, 1 => 同一人物
                if person_list[i] != person_list[j]:
                    match_true.append(0)
                else:
                    match_true.append(1)
        scores[th] = f1_score(match_true, match_pred)
    # スコアが高い順にソート
    result = sorted(list(scores.items()), key=lambda x: x[1])[::-1]
    return result


if __name__ == "__main__":
    result = evaluate()
    best_th, best_score = result[0]
    print(result)
    print(f"best score: {best_score}, threathold: {best_th:.2f}")
