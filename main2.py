import os

from captcha.image import ImageCaptcha
from random import randint
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# 10数字+26大写字母+26小写字母
list = [chr(i) for i in range(48, 58)] + [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]

def gen_train_captcha(num, captcha_len):
    for j in range(num):
        if j % 100 == 0:
            print(j)
        chars = ''.join(random.choices(list, k=captcha_len))
        image = ImageCaptcha().generate_image(chars)
        image.save('./train_img/' + chars + '.jpg')

def gen_test_captcha(num, captcha_len):
    for j in range(num):
        if j % 100 == 0:
            print(j)
        chars = ''.join(random.choices(list, k=captcha_len))
        image = ImageCaptcha().generate_image(chars)
        image.save('./test_img/' + chars + '.jpg')

def gen_validation_captcha(num, captcha_len):
    for j in range(num):
        if j % 100 == 0:
            print(j)
        chars = ''.join(random.choices(list, k=captcha_len))
        image = ImageCaptcha().generate_image(chars)
        image.save('./validation_img/' + chars + '.jpg')


import network
# import predict
gen_train_captcha(500000, 4)
gen_test_captcha(50000, 4)
gen_validation_captcha(50000, 4)
