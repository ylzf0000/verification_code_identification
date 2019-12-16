import os

from captcha.image import ImageCaptcha
from random import randint
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def gen_captcha(num, captcha_len):
    # # 纯数字，如果目标识别是10位数字，预训练用纯数字效果更好
    # list = [chr(i) for i in range(48, 58)]

    # # 10数字+26大写字母+26小写字母
    list = [chr(i) for i in range(48, 58)] + [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]

    for j in range(num):
        if j % 100 == 0:
            print(j)
        chars = ''.join(random.choices(list, k=captcha_len))
        # chars = ''
        # for i in range(captcha_len):
        #     rand_num = randint(0, 61)
        #     chars += list[rand_num]
        image = ImageCaptcha().generate_image(chars)
        image.save('./train_img/' + chars + '.jpg')


# num = 50000
# captcha_len = 4
# gen_captcha(num, captcha_len)

import network
# import predict