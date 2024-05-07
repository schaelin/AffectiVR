import glob
import os
import pickle

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from src.data.generator import Data_Generator
from model import train_model


# learning rate scheduler 정의
def lr_schedule(epoch):
    lr = 0.001
    if epoch > 5: lr /= 10
    if epoch > 10: lr /= 10
    if epoch > 15: lr /= 10
    if epoch > 20: lr /= 10
    if epoch > 25: lr /= 10
    return lr


def train(model_function, feature_type, loss, data_path, save_path, hist_path, input_shape, batch_size, epochs):
    # Load model
    model = model_function(input_shape=input_shape, feature_type=feature_type)
    model.summary()

    # Load data
    gen_train = Data_Generator(data_path, Data_Generator.DATA_TYPE_TOTAL_TRAIN, batch_size, shuffle=True)
    gen_val = Data_Generator(data_path, Data_Generator.DATA_TYPE_TOTAL_VALIDATION, batch_size, shuffle=True)

    # Set callbacks
    scheduler = LearningRateScheduler(lr_schedule)
    check_point = ModelCheckpoint(filepath=save_path[:-3] + '_best.h5', verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=True)
    callbacks = [check_point, scheduler, early_stopping]

    # Train
    model.compile(optimizer=Adam(lr=0.001), loss=loss)
    with tf.device("/device:GPU:0"):
        history = model.fit_generator(generator=gen_train,
                                      epochs=epochs,
                                      validation_data=gen_val,
                                      callbacks=callbacks)

    # Save result
    model.save(save_path, include_optimizer=False)
    with open(hist_path, 'wb') as file:
        pickle.dump(history.history, file)

    # Show result
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()


if __name__ == '__main__':
    model_function = train_model
    feature_type = 'deep_resnet'
    loss = 'binary_crossentropy'

    input_shape = (240, 320, 1)
    batch_size = 16
    epochs = 10

    datas = sorted(glob.glob('../../../data/pupil_dataset/data.h5'))
    # k-fold 시작
    for data in datas:
        print(f'fold {data[-4]} starting...')

        save_path = '../../../result/result_105_18_' + data[-4] + '/model.h5'
        hist_path = '../../../result/result_105_18_' + data[-4] + '/history'

        # model 저장 경로 없을 경우 생성
        if not os.path.isdir(save_path[:-9]):
            os.mkdir(save_path[:-9])

        train(model_function, feature_type, loss, data, save_path, hist_path, input_shape, batch_size, epochs)
