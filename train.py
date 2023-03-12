import numpy as np
from tqdm import tqdm
import os
from utils import create_explicit_dataset, create_transition_pattern_dataset
from model import Model
import warnings
import tensorflow as tf
from tensorflow import keras
from evaluate import *
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')


def train(model: tf.keras.Model, input, true_output, optimizer):
    with tf.GradientTape() as tape:
        pred_output = model(input)
        current_loss = tf.reduce_mean(keras.losses.mean_squared_error(true_output, pred_output))
    grads = tape.gradient(current_loss, model.variables)
    grads_and_variables = zip(grads, model.variables)
    optimizer.apply_gradients(grads_and_variables)
    return current_loss.numpy()


def gen_random_batch(x_train, y_train, batch_size):
    idx = 0
    while True:
        if idx + batch_size > len(x_train):
            idx = 0
        start = idx
        idx += batch_size
        yield x_train[start:start + batch_size], y_train[start:start + batch_size]


def eval(feature_column, train_data, test_data, pred_rating_matrix):
    trainSet = {}
    testSet_5 = {}
    testSet = {}
    for line in train_data:
        user, movie = line[0], line[1]
        trainSet.setdefault(user, set())
        trainSet[user].add(movie)
    for line in test_data:
        user, movie, rating = line[0], line[1], line[2]
        if rating > 4:
            testSet_5.setdefault(user, set())
            testSet_5[user].add(movie)
    for line in test_data:
        user, movie, rating = line[0], line[1], line[2]
        testSet.setdefault(user, set())
        testSet[user].add(movie)
    rec(feature_column, trainSet, testSet, [10, 20, 30, 40, 50, 60], pred_rating_matrix)
    hr(feature_column, trainSet, testSet, [5, 10, 15, 20, 25, 30], pred_rating_matrix)


if __name__ == "__main__":
    # ------ 读入数据 ------ #
    dataset_path = '../dataset/ml-100k/'
    f = open('./res/history.his', 'w')
    now = time.time()
    for i in [0.2]:
        f.write('test_size:%.2f\n' % i)
        feat_column, train_data, test_data, user_data = create_explicit_dataset(dataset_path + 'ratings.csv',
                                                                                dataset_path + 'u.user', 16, i)

        user_num = feat_column[0]['feat_num']
        item_num = feat_column[1]['feat_num']
        tp_data = create_transition_pattern_dataset(train_data, item_num)

        # ------ 训练模型 ------ #
        # ------ 参数设置 ------ #
        n_latent_factors = 16
        epochs = 20
        batch_size = 32
        steps_per_epoch = len(train_data) // batch_size
        learning_rate = 0.001
        # 训练用户-项目模型
        model = Model(user_num, item_num, n_latent_factors)
        explict_MF = model.Explict_MF(user_data)
        implicit_MF = model.Implicit_MF()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        gen = gen_random_batch(train_data[:, 0:2], train_data[:, 2], batch_size)
        # for epoch in range(epochs):
        #     for step in tqdm(range(steps_per_epoch)):
        #         x_batch, y_batch = next(gen)
        #         # ------ 手动训练 ------ #
        #         loss = train(explict_MF, x_batch, y_batch, optimizer)
        #         # print("loss:%s" % loss)
        #     pred_output = explict_MF.predict(test_data[:, 0:2])
        #     rmse = tf.sqrt(tf.reduce_mean(keras.losses.mean_squared_error(test_data[:, 2], pred_output))).numpy()
        #     print("epoch:%d\trmse:%s" % (epoch + 1, rmse))
        #     ------ 自动训练 ------ #
        for epoch in range(epochs):
            print("Epoch\t%d/%d" % (epoch + 1, epochs))
            explict_MF.compile(optimizer, loss='mse', metrics=[keras.metrics.RootMeanSquaredError()])
            implicit_MF.compile(optimizer, loss='mse')
            implicit_MF.fit(
                tp_data[0],
                tp_data[1],
                batch_size,
                1,
            )
            hist = explict_MF.fit(
                train_data[:, 0:2],
                train_data[:, 2],
                batch_size,
                1,
                validation_data=(test_data[:, 0:2], test_data[:, 2]),
            )
            f.write('val_root_mean_squared_error:%s\n' %
                    hist.history['val_root_mean_squared_error'][0])
        # ------ 评估 ------ #
        rmse = explict_MF.evaluate(test_data[:, 0:2], test_data[:, 2])[1]
        print("rmse:%.6f" % rmse)
        explict_MF.save_weights('./res/weights/ml-100k/')
        # ------ 进行推荐 ------ #
        # 给用户1推荐top10
        user = 1
        predict_list = np.concatenate([np.reshape(np.tile(np.reshape(np.arange(user_num), (-1, 1)), item_num), (-1, 1)),
                                       np.reshape(np.tile(np.arange(item_num), user_num), (-1, 1))], axis=1)
        R = np.reshape(explict_MF.predict(predict_list, batch_size=8192,
                                          verbose=1), (user_num, item_num))
        eval(feat_column, train_data, test_data, R)
    f.close()
    print("time:%s" % (time.time() - now))
