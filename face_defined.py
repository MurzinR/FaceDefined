import face_recognition
import numpy as np
from scipy.optimize import minimize


def get_face_defender():
    print('Обучение нейронной сети:')
    train_i = []
    for i in range(1, 12):
        train_i.extend(face_recognition.face_encodings(face_recognition.load_image_file(f"i_{i}.jpg")))
    train_not_i = face_recognition.face_encodings(face_recognition.load_image_file("train_not_i.jpg"))
    train_i.extend(train_not_i)
    face_encodings = face_recognition.face_encodings(face_recognition.load_image_file(f"2020-03-11-214159.jpg"))[0]
    # входные данные
    x = np.array([np.linalg.norm(face_encodings - face_to_compare) for face_to_compare in train_i])
    print(x)
    # выходные данные
    y = np.array([1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    def error(k):
        print(((k[2] * x ** 2 + x * k[1] + k[0] - y) ** 2).sum(), k)
        return ((k[2] * x ** 2 + x * k[1] + k[0] - y) ** 2).sum()

    k_opt = minimize(error, np.array([0.5, 0.5, 0.5]), method='BFGS').x

    def result(ex):
        return 1 / (1 + np.exp(-(k_opt[2] * ex ** 2 + ex * k_opt[1] + k_opt[0])))

    return result


if __name__ == '__main__':
    print(get_face_defender()(0.6))
