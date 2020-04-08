import os
import numpy as np
import glob
from tensorflow import keras
from scipy import misc
import imageio
from generate_captcha import label_list, captcha_len, label_map

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

model_filepath = './captcha_0aA.h5'
img_size = (60, 160, 3)


def build_network():
    input_image = keras.layers.Input(shape=img_size)
    base_model = keras.applications.xception.Xception(input_tensor=input_image,
                                                      weights='imagenet',
                                                      include_top=False,
                                                      pooling='max')

    outputs = []
    for _ in range(captcha_len):
        x = base_model.output
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(len(label_list), activation='softmax')(x)
        outputs.append(x)
    model = keras.models.Model(inputs=input_image, outputs=outputs)
    if os.path.exists(model_filepath):
        model.load_weights(model_filepath)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def plot_model(model):
    keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


if __name__ == '__main__':
    model = build_network()
    plot_model(model)
