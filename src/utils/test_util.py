import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model


def train_model_to_test_model(model_path, test_model_func, feature_type=None):
    train_model = load_model(model_path, compile=False, custom_objects={'K': K})
    train_model.get_layer('backbone').summary()

    model = test_model_func(feature_type=feature_type)
    model.get_layer('backbone').summary()

    model.get_layer('backbone').set_weights(train_model.get_layer('backbone').get_weights())
    return model


# predict 후 결과 npy file 저장
def test(model_path, test_model_func, gen, feature_type=None, save_path='../../result/result.npy'):
    # Load model
    model = train_model_to_test_model(model_path, test_model_func, feature_type)

    y = gen.labels
    p = model.predict_generator(gen, gen.__len__(), verbose=1)

    p = p.flatten()
    y = y.flatten()[:p.shape[0]]

    pred_data = np.array([y, p])
    np.save(save_path, pred_data)


# prediction 결과 분석 (EER, FRR, roc, GI 붚포, confusion, threshold)
def show_test_result(path, is_fold=False):
    res = np.load(path)

    y, p = res
    genuine = p[y == 1.0]
    imposter = p[y == 0.0]

    eer_dif = 100
    eer_thres = 0
    eer_tp, eer_fp, eer_fn, eer_tn = 0, 0, 0, 0

    csv_path = path[:path.rfind('/')]+'/csv'+path[path.rfind('/'):-4]+'.csv'
    file = open(csv_path, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(['thres', 'tp', 'fn', 'fp', 'tn', 'far(fpr)', 'frr', 'fpr'])

    fars = []
    tprs = []

    # 1000단위로 threshold 나눠서 계산
    for thres in np.arange(np.min(p), np.max(p), (np.max(p) - np.min(p)) / 1000.0):
        tp = np.sum(genuine < thres, dtype=np.int)
        fn = np.sum(genuine >= thres, dtype=np.int)
        fp = np.sum(imposter < thres, dtype=np.int)
        tn = np.sum(imposter >= thres, dtype=np.int)

        far = fp / imposter.shape[0]
        frr = fn / genuine.shape[0]
        tpr = tp / genuine.shape[0]

        fars.append(far)
        tprs.append(tpr)

        writer.writerow([thres, tp, fn, fp, tn, far, frr, 1-frr])

        dif = abs(far - frr)
        if dif < eer_dif:
            eer_thres = thres
            eer_dif = dif
            eer_tp, eer_fp, eer_fn, eer_tn = tp, fp, fn, tn
    file.close()

    zero_far_thres = np.min(imposter)
    zero_far_fn = np.sum(genuine > zero_far_thres, dtype=np.int)

    far_rate = 0.01  # Error 1%
    idx = int(imposter.shape[0] * far_rate)
    sorted_imposter = np.sort(imposter)
    thres = sorted_imposter[idx - 1]
    far_lt_1_fn = np.sum(genuine > thres, dtype=np.int)

    far_rate = 0.001  # Error 0.1%
    idx = int(imposter.shape[0] * far_rate)
    sorted_imposter = np.sort(imposter)
    thres = sorted_imposter[idx - 1]
    far_lt_0_1_fn = np.sum(genuine > thres, dtype=np.int)

    far_rate = 0.0001  # Error 0.01%
    idx = int(imposter.shape[0] * far_rate)
    sorted_imposter = np.sort(imposter)
    thres = sorted_imposter[idx - 1]
    far_lt_0_01_fn = np.sum(genuine > thres, dtype=np.int)

    print('total data ', y.shape[0])
    print('genuine data ', genuine.shape[0])
    print('imposter data ', imposter.shape[0])
    print()

    print('Confusion matrix')
    print('TP      FN')
    print('FP      TN')
    print(eer_tp, '    ', eer_fn)
    print(eer_fp, '    ', eer_tn)
    print()

    print('EER threshold ', eer_thres)
    print('genuine min / max ', np.min(genuine), np.max(genuine))
    print('imposter min / max ', np.min(imposter), np.max(imposter))
    print()

    print('FAR : ', eer_fp / imposter.shape[0] * 100)
    print('FRR : ', eer_fn / genuine.shape[0] * 100)
    print('EER : ', (eer_fp+eer_fn) / (imposter.shape[0]+genuine.shape[0]) * 100)
    print('FRR when FAR is less than 1% : ', far_lt_1_fn / genuine.shape[0] * 100)
    print('FRR when FAR is less than 0.1% : ', far_lt_0_1_fn / genuine.shape[0] * 100)
    print('FRR when FAR is less than 0.01% : ', far_lt_0_01_fn / genuine.shape[0] * 100)
    print('FRR when FAR is zero : ', zero_far_fn / genuine.shape[0] * 100)

    # show genuine imposter distribution
    plt.hist(genuine, alpha=0.6, bins=100, color='r', label='genuine', density=True)
    plt.hist(imposter, alpha=0.6, bins=100, color='b', label='imposter', density=True)
    plt.legend()
    plt.show()

    roc_auc = auc(fars, tprs)

    # show ROC curve
    plt.rc('font', size=14)
    plt.figure(figsize=(8, 8))
    plt.plot(fars, tprs, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.show()

    # k-fold 적용할 경우 평균 계산을 위해 EER, FRR return
    if is_fold:
        return (eer_fp+eer_fn) / (imposter.shape[0]+genuine.shape[0]) * 100, far_lt_1_fn / genuine.shape[0] * 100
