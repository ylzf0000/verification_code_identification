import os
import numpy as np
import glob
from tensorflow import keras
from scipy import misc
import imageio
from generate_captcha import label_list, captcha_len, label_map
from network import model_filepath, img_size, build_network

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"


train_samples = glob.glob('./train_img/*.jpg')
np.random.shuffle(train_samples)
validation_samples = glob.glob('./validation_img/*.jpg')
np.random.shuffle(validation_samples)


def train(model: keras.models.Model):
    try:
        model.fit_generator(
            generator=data_generator(train_samples, 128),
            steps_per_epoch=128,
            epochs=10,
            verbose=1,
            validation_data=data_generator(validation_samples, 128),
            validation_steps=128,
        )
        model.save_weights(model_filepath)
    except KeyboardInterrupt as error:
        model.save_weights(model_filepath)
        print(error)


def data_generator(data, batch_size):
    while True:
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


if __name__ == '__main__':
    model = build_network()
    train(model)