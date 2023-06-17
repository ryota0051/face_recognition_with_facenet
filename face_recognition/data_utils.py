from pathlib import Path
from typing import Tuple

from PIL import Image


def open_person_img(img_path: str) -> Tuple[Image.Image, str]:
    # 画像の読み込み
    img = open_and_convert_img2RGB(img_path)
    # 画像パスから人名を取得
    img_name = Path(img_path).stem
    person_name = img_name[: len(img_name) - 1]
    return (img, person_name)


def open_and_convert_img2RGB(img_path: str) -> Image.Image:
    img = Image.open(img_path)
    img = img.convert("RGB")
    return img
