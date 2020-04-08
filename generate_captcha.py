import os
from captcha.image import ImageCaptcha
import random
import uuid

# 10数字+26大写字母+26小写字母
label_list = [str(x) for x in range(10)] \
             + [chr(i) for i in range(ord('a'), ord('z') + 1)] \
             + [chr(i) for i in range(ord('A'), ord('Z') + 1)]
label_map = {w: i for i, w in enumerate(label_list)}
captcha_len = 4
train_img_dir = './train_img/'


def gen_captcha(num, dir, captcha_len=captcha_len):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for i in range(num):
        if i % 1000 == 0:
            print(i)
        chars = ''.join(random.choices(label_list, k=captcha_len))
        id = str(uuid.uuid4()).replace('-', '')
        file_full_path = f'{dir}{id}_{chars}.jpg'
        image = ImageCaptcha().generate_image(chars)
        image.save(file_full_path)


def gen_train_captcha(num=51200, captcha_len=captcha_len):
    dir = './train_img/'
    gen_captcha(num, dir, captcha_len)


def gen_validation_captcha(num=10240, captcha_len=captcha_len):
    dir = './validation_img/'
    gen_captcha(num, dir, captcha_len)


def gen_test_captcha(num=10240, captcha_len=captcha_len):
    dir = './test_img/'
    gen_captcha(num, dir, captcha_len)


if __name__ == '__main__':
    gen_train_captcha(51200)
    gen_validation_captcha(10240)
    gen_test_captcha(10240)
