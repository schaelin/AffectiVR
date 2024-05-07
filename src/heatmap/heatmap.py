import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from src.data.generator import Data_Generator
from src.models.proposed.model import *
from src.models.proposed.model import test_model_for_heatmap

tf.compat.v1.disable_eager_execution()


def train_model_to_heatmap_model(model_path, test_model_func, feature_type=None):
    trained_model = train_model(feature_type=feature_type)
    trained_model.load_weights(model_path)
    model = test_model_func(feature_type=feature_type)

    # heatmap model의 모든 conv, batch, dense layer name 저장
    target_conv = []
    target_bn = []
    target_dense = []
    for layer in model.layers:
        if 'model' in layer.name:
            for l in layer.layers:
                if 'conv' in l.name:
                    target_conv.append(l)
                elif 'batch' in l.name:
                    target_bn.append(l)
                elif 'dense' in l.name:
                    target_dense.append(l)
        else:
            if 'conv' in layer.name:
                target_conv.append(layer)
            elif 'batch' in layer.name:
                target_bn.append(layer)
            elif 'dense' in layer.name:
                target_dense.append(layer)

    # heatmap model의 모든 conv, batch, dense layer에 weight load
    train_conv = []
    train_bn = []
    train_dense = []
    for layer in trained_model.layers:

        if 'model' in layer.name or 'backbone' in layer.name:
            for l in layer.layers:
                if 'conv' in l.name:
                    train_conv.append(l)
                elif 'batch' in l.name:
                    train_bn.append(l)
                elif 'dense' in l.name:
                    train_dense.append(l)
        else:
            if 'conv' in layer.name:
                train_conv.append(layer)
            elif 'batch' in layer.name:
                train_bn.append(layer)
            elif 'dense' in layer.name:
                train_dense.append(layer)

    for target, train in zip(target_conv, train_conv):
        target.set_weights(train.get_weights())
    for target, train in zip(target_bn, train_bn):
        target.set_weights(train.get_weights())
    for target, train in zip(target_dense, train_dense):
        target.set_weights(train.get_weights())

    model.save(model_path[:-3]+'_heatmap.h5')


def get_heatmap_image(grad, map):
    # grad mean 기준 map pixel에 weight 적용
    weight = np.mean(grad, axis=(0, 1))
    hmap = weight * map

    # color normalization
    hmap = np.mean(hmap, axis=-1)
    hmap = np.maximum(hmap, 0)
    hmap = (hmap / hmap.max()) * 255
    hmap = hmap.astype(np.uint8)
    hmap = cv2.resize(hmap, (320, 240))
    hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    return hmap


def stack_images(stack, img, x, y, size=(240, 320)):
    sy = y * size[0]
    ey = sy + size[0]
    sx = x * size[1]
    ex = sx + size[1]
    stack[sy:ey, sx:ex] = img[:, :]
    return stack


def get_heatmap(model_path, gen, thres=None):
    # Load model
    model = load_model(model_path, compile=False)
    model.summary()

    model_input = model.inputs
    model_output = model.output

    names = ['for_gradient_1_1', 'for_gradient_1_2', 'for_gradient_1_3', 'for_gradient_1_4',
             'for_gradient_2_1', 'for_gradient_2_2', 'for_gradient_2_3', 'for_gradient_2_4',
             'for_vector_1_1', 'for_vector_1_2', 'for_vector_1_3', 'for_vector_1_4',
             'for_vector_2_1', 'for_vector_2_2', 'for_vector_2_3', 'for_vector_2_4',
             'for_network_1_1', 'for_network_1_2', 'for_network_1_3',
             'for_network_2_1', 'for_network_2_2', 'for_network_2_3']

    targets = []
    for name in names:
        targets.append(model.get_layer(name).output)

    grads = []
    for t in targets:
        grad = K.gradients(model_output, t)[0]
        grads.append(grad)

    # 각 layer 별 gradient와 output 반환하는 함수 정의
    heatmap_function = K.function(model_input, targets[:8] + grads + [model_output])

    key = 0
    for i in range(gen.__len__()):
        xs, ys = gen.__getitem__(i)
        res = heatmap_function(xs)

        map1 = res[0: 4]
        map2 = res[4: 8]
        grad1 = res[8: 12]
        grad2 = res[12: 16]
        vec1 = res[16: 20]
        vec2 = res[20: 24]
        net1 = res[24: 27]
        net2 = res[27: 30]
        pred = res[-1]

        for j, (x1, x2, y, p) in enumerate(zip(xs[0], xs[1], ys, pred)):
            if y[0] == 0:
                print('Label : Imposter\t/\t', end='')
            else:
                print('Label : Genuine\t/\t', end='')

            if p[0] < thres:
                print('Pred : Genuine\t', end='')
            else:
                print('Pred : Imposter\t', end='')

            print('Pred : ', p[0], '\tThreshold : ', thres)

            x1 = (x1 * 255).astype(np.uint8)
            x2 = (x2 * 255).astype(np.uint8)

            # heatmap initiation
            heatmap_stack1 = np.zeros((240 * 3, 320*len(grad1), 3), np.uint8)
            heatmap_stack2 = np.zeros((240 * 3, 320*len(grad2), 3), np.uint8)

            # gradients heatmap
            for idx, (g1, g2, m1, m2) in enumerate(zip(grad1, grad2, map1, map2)):
                img1 = get_heatmap_image(g1[j], m1[j])
                img2 = get_heatmap_image(-g2[j], m2[j])

                img1 = img1//2 + x1//2
                img2 = img2//2 + x2//2

                heatmap_stack1 = stack_images(heatmap_stack1, img1, idx, 0)
                heatmap_stack2 = stack_images(heatmap_stack2, img2, idx, 0)

            # vector heatmap
            for idx, (v1, v2, m1, m2) in enumerate(zip(vec1, vec2, map1, map2)):
                img1 = get_heatmap_image(v1[j], m1[j])
                img2 = get_heatmap_image(-v2[j], m2[j])
                img1 = img1//2 + x1//2
                img2 = img2//2 + x2//2
                heatmap_stack1 = stack_images(heatmap_stack1, img1, idx, 1)
                heatmap_stack2 = stack_images(heatmap_stack2, img2, idx, 1)

            # network output heatmap
            for idx, (n1, n2, m1, m2) in enumerate(zip(net1, net2, map1, map2)):
                img1 = get_heatmap_image(n1[j], m1[j])
                img2 = get_heatmap_image(-n2[j], m2[j])
                img1 = img1//2 + x1//2
                img2 = img2//2 + x2//2
                heatmap_stack1 = stack_images(heatmap_stack1, img1, idx, 2)
                heatmap_stack2 = stack_images(heatmap_stack2, img2, idx, 2)

            # input stack
            heatmap_stack1 = stack_images(heatmap_stack1, x1, 0, 1)
            heatmap_stack2 = stack_images(heatmap_stack2, x2, 0, 1)

            # small heatmap
            heatmap_stack11 = heatmap_stack1[:480, :320, :]
            heatmap_stack22 = heatmap_stack2[:480, :320, :]

            heatmap_stack_fin = np.zeros((480, 640, 3), np.uint8)
            heatmap_stack_fin[:, :320, :] = heatmap_stack11
            heatmap_stack_fin[:, 320:, :] = heatmap_stack22

            cv2.imshow('heatmap fin', heatmap_stack_fin)

            cv2.imshow('heatmap 1', heatmap_stack1)
            cv2.imshow('heatmap 2', heatmap_stack2)

            key = cv2.waitKey(0)
            if key == ord('s'):
                cv2.imwrite('heatmap.jpg', heatmap_stack_fin)
            if key == ord('q'):
                break
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    data_path = '../../data/pupil_dataset/data.h5'
    batch_size = 16

    data = Data_Generator(data_path, Data_Generator.DATA_TYPE_TOTAL_TEST, batch_size, False)

    # Show heatmap
    thres = {'resnet34': 0.58, 'senet': 0.58, 'deep_resnet': 0.58}

    feature_type = 'deep_resnet'
    model_path = '../../result/model_best.h5'
    save_path = ''
    train_model_to_heatmap_model(model_path, test_model_for_heatmap, feature_type)

    model_path = '../../result/model_best_heatmap.h5'
    get_heatmap(model_path=model_path, gen=data, thres=thres[feature_type])
