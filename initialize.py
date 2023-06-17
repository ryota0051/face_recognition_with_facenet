from facenet_pytorch import MTCNN, InceptionResnetV1

model = InceptionResnetV1(pretrained="vggface2").eval()
mtcnn = MTCNN()
