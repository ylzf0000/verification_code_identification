import os

import imageio
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
import glob
from generate_captcha import label_list, captcha_len, label_map
from train import img_size
import network

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

test_samples = glob.glob(r'./test_img/*.jpg')
np.random.shuffle(test_samples)
batch = None


def data_generator_test(data, batch_size):
    while True:
        global batch
        batch = np.random.choice(data, batch_size)
        X, Y = [], []
        for img in batch:
            X.append(np.array(imageio.imread(img)).reshape(img_size))
            real_label = img[-8:-4]
            y_list = [label_map[w] for w in real_label]
            Y.append(y_list)

        X = keras.applications.xception.preprocess_input(np.array(X).astype(float))
        Y = np.array(Y)
        yield X, [Y[:, i] for i in range(4)]


def predict(model, batch_size, plot=False):
    x, y = next(data_generator_test(test_samples, batch_size))
    z = model.predict(x)
    z = np.array(z)
    y = np.array(y)
    w = np.zeros(shape=(captcha_len, batch_size), dtype=int)
    for i in range(z.shape[0]):
        w[i] = [np.array(z[i][j]).argmax() for j in range(z.shape[1])]

    y_predicted_str_list = [''.join([label_list[w[i][j]] for i in range(w.shape[0])]) for j in range(w.shape[1])]
    y_true_str_list = [''.join([label_list[y[i][j]] for i in range(y.shape[0])]) for j in range(y.shape[1])]

    num_right = sum([y_true_str_list[i] == y_predicted_str_list[i] for i in range(batch_size)])
    if plot:
        print(f'正确率: {num_right / batch_size * 100}%')
    for i in range(batch_size):
        y_true, y_predicted = y_true_str_list[i], y_predicted_str_list[i]
        b = (y_true == y_predicted)
        if not b:
            print(f'true: {y_true}, predicted: {y_predicted}')
            if plot:
                image = mpimg.imread(batch[i])
                plt.axis('off')
                plt.imshow(image)
                plt.show()
    if not plot:
        print(f'正确率: {num_right / batch_size * 100}%')
    return batch_size, num_right


if __name__ == '__main__':
    model = network.build_network()
    ns, nrs = 0, 0
    for i in range(10):
        n, nr = predict(model, 128, True)
        ns += n
        nrs += nr
    print(f'平均10次正确率：{nrs / ns * 100}%')

    # for j in range(z.shape[1]):
    #     w[i][j] = np.array(z[i][j]).argmax()

    # for j in range(w.shape[1]):
    #     y_predicted_str_list.append(''.join([label_list[w[i][j]] for i in range(w.shape[0])]))
    # for j in range(y.shape[1]):
    #     y_true_str_list.append(''.join([label_list[y[i][j]] for i in range(y.shape[0])]))