import numpy as np
import glob
from keras.applications.xception import Xception, preprocess_input
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from scipy import misc
samples = glob.glob('./train_img/*.jpg')  # 获取所有样本图片
np.random.shuffle(samples)  # 将图片打乱

nb_train = 45000  # 共有5万样本，4.5万用于训练，5k用于验证
train_samples = samples[:nb_train]
test_samples = samples[nb_train:]

letter_list = [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]  # 需要识别的36类

# CNN适合在高宽都是偶数的情况，否则需要在边缘补齐，把全体图片都resize成这个尺寸(高，宽，通道)
img_size = (60, 160)
input_image = Input(shape=(img_size[0], img_size[1], 3))

# 直接将验证码输入，做几个卷积层提取特征，然后把这些提出来的特征连接几个分类器（36分类，因为不区分大小写），
# 输入图片
# 用预训练的Xception提取特征,采用平均池化
base_model = Xception(input_tensor=input_image, weights='imagenet', include_top=False, pooling='avg')

# 用全连接层把图片特征接上softmax然后36分类，dropout为0.5，因为是多分类问题，激活函数使用softmax。
# ReLU - 用于隐层神经元输出
# Sigmoid - 用于隐层神经元输出
# Softmax - 用于多分类神经网络输出
# Linear - 用于回归神经网络输出（或二分类问题）
predicts = [Dense(36, activation='softmax')(Dropout(0.5)(base_model.output)) for i in range(4)]

model = Model(inputs=input_image, outputs=predicts)


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.load_weights('CaptchaForPython.h5')

# misc.imread把图片转化成矩阵，
# misc.imresize重塑图片尺寸misc.imresize(misc.imread(img), img_size)  img_size是自己设定的尺寸
# ord()函数主要用来返回对应字符的ascii码，
# chr()主要用来表示ascii码对应的字符他的输入时数字，可以用十进制，也可以用十六进制。

def data_generator(data, batch_size):  # 样本生成器，节省内存
    while True:
        # np.random.choice(x,y)生成一个从x中抽取的随机数,维度为y的向量，y为抽取次数
        batch = np.random.choice(data, batch_size)
        x, y = [], []
        for img in batch:
            x.append(misc.imresize(misc.imread(img), img_size))  # 读取resize图片,再存进x列表
            real_num = img[-8:-4]
            y_list = []
            for i in real_num:
                i = i.upper()
                if ord(i) - ord('A') >= 0:
                    y_list.append(ord(i) - ord('A') + 10)
                else:
                    y_list.append(ord(i) - ord('0'))

            y.append(y_list)

        # 把验证码标签添加到y列表,ord(i)-ord('a')把对应字母转化为数字a=0，b=1……z=26
        x = preprocess_input(np.array(x).astype(float))
        # 原先是dtype=uint8转成一个纯数字的array
        y = np.array(y)
        yield x, [y[:, i] for i in range(4)]
        # 输出：图片array和四个转化成数字的字母 例如：[array([6]), array([0]), array([3]), array([24])])


model.fit_generator(data_generator(train_samples, 100), steps_per_epoch=1050, epochs=10,
                    validation_data=data_generator(test_samples, 100), validation_steps=100)
# 参数：generator生成器函数,
# samples_per_epoch，每个epoch以经过模型的样本数达到samples_per_epoch时，记一个epoch结束
# step_per_epoch:整数，当生成器返回step_per_epoch次数据是记一个epoch结束，执行下一个epoch
# epochs:整数，数据迭代的轮数
# validation_data三种形式之一，生成器，类（inputs,targets）的元组，或者（inputs,targets，sample_weights）的元祖
# 若validation_data为生成器，validation_steps参数代表验证集生成器返回次数
# class_weight：规定类别权重的字典，将类别映射为权重，常用于处理样本不均衡问题。
# sample_weight：权值的numpy array，用于在训练时调整损失函数（仅用于训练）。可以传递一个1D的与样本等长的向量用于对样本进行1对1的加权，或者在面对时序数据时，传递一个的形式为（samples，sequence_length）的矩阵来为每个时间步上的样本赋不同的权。这种情况下请确定在编译模型时添加了sample_weight_mode='temporal'。
# workers：最大进程数
# max_q_size：生成器队列的最大容量
# pickle_safe: 若为真，则使用基于进程的线程。由于该实现依赖多进程，不能传递non picklable（无法被pickle序列化）的参数到生成器中，因为无法轻易将它们传入子进程中。
# initial_epoch: 从该参数指定的epoch开始训练，在继续之前的训练时有用。

# 保存模型
model.save('CaptchaForPython.h5')