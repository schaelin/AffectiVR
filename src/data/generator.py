import numpy as np
import cv2
import h5py
import tensorflow


class Generator(tensorflow.keras.utils.Sequence):
    DATA_TYPE_TOTAL = ''
    DATA_TYPE_TOTAL_TRAIN = 'train_'
    DATA_TYPE_TOTAL_VALIDATION = 'validation_'
    DATA_TYPE_TOTAL_TEST = 'test_'

    def __init__(self, path, data_type, batch_size, shuffle):
        self.path = path
        self.data_type = data_type
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.images = h5py.File(path, 'r')['images']
        self.index = h5py.File(path, 'r')[data_type + 'index']
        self.count = h5py.File(path, 'r')[data_type + 'count']

        self.on_epoch_end()

    def on_epoch_end(self):
        pairs = []
        labels = []

        if 'test' in self.data_type:
            # test
            for index_idx, (idx, cnt) in enumerate(zip(self.index, self.count)):
                for i in range(cnt):
                    # genuine
                    for j in range(i + 1, cnt):
                        pairs.append([idx + i, idx + j])
                        labels.append(1)
                    # imposter
                    for im_idx, im_cnt in zip(self.index[index_idx + 1:], self.count[index_idx + 1:]):
                        # genuine idx 뒤부터 모든 idx에 대해 imposter pair로 추가
                        for j in range(im_cnt):
                            pairs.append([idx + i, im_idx + j])
                            labels.append(0)
        else:
            # train, validation
            for index_idx, (idx, cnt) in enumerate(zip(self.index, self.count)):
                count_list = np.array(list(range(cnt)))
                np.random.shuffle(count_list)
                count_list = np.append(count_list, count_list[:cnt // 2], axis=-1)

                for i, anchor in enumerate(count_list[:cnt]):
                    anchor_index = idx + anchor
                    for positive in count_list[i + 1: i + 1 + cnt // 2]:
                        positive_index = idx + positive

                        rand_idx = np.random.randint(0, len(self.index) - 1)
                        # negative sample idx 겹침 방지
                        if rand_idx >= index_idx:
                            rand_idx += 1
                        negative_index = self.index[rand_idx] + np.random.randint(0, self.count[rand_idx])

                        pairs.append([anchor_index, positive_index])
                        labels.append(1)
                        pairs.append([anchor_index, negative_index])
                        labels.append(0)

        self.pairs = np.array(pairs)
        self.labels = np.expand_dims(np.array(labels), axis=-1)

        if self.shuffle:
            idx = np.random.permutation(self.labels.shape[0])
            self.pairs = self.pairs[idx]
            self.labels = self.labels[idx]

        print('Class : ', len(self.index))
        print('Total image pair : ', len(self.pairs))

    def __len__(self):
        return len(self.labels) // self.batch_size


class Data_Generator(Generator):
    def __init__(self, path, data_type, batch_size, shuffle):
        super().__init__(path, data_type, batch_size, shuffle)
    
    # 밝기, 이동 augmentation 함수
    def augmentation(self, x, scale, scale_factor, bright, bright_factor):
        if scale:
            x = cv2.resize(x, dsize=None, fx=scale_factor, fy=scale_factor)
            sx, sy = np.random.randint(0, x.shape[1] - 320), np.random.randint(0, x.shape[0] - 240)
            x = x[sy: sy + 240, sx: sx + 320]
            x = np.expand_dims(x, axis=-1)
        if bright:
            x = x.astype(np.float32) + bright_factor
        return x

    def __getitem__(self, batch_index):
        # test 과정엔 augmentation 적용 X
        is_augment = 'train' in self.data_type or 'validation' in self.data_type

        # Get labels
        ys = self.labels[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
        pair_list = self.pairs[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]

        # uniform 분포에 따라 scale, bright augmentation 진행 여부 결정 및 factor value 적용
        scale = np.random.uniform(0, 1) < 0.5
        scale_factor = np.random.uniform(1.1, 1.3)
        bright = np.random.uniform(0, 1) < 0.5
        bright_factor = np.random.uniform(-10, 10)

        x = [[], [], []]
        for idxs in pair_list:
            for i, idx in enumerate(idxs):
                img = self.images[idx]
                if is_augment:
                    img = self.augmentation(img, scale, scale_factor, bright, bright_factor)
                x[i].append(img)

        # 모델에 input으로 넣기 위해 이미지 화소값 범위 0~1 정규화
        x1 = np.array(x[0], np.float32) / 255
        x2 = np.array(x[1], np.float32) / 255
        xs = [x1, x2]

        return xs, ys


if __name__ == '__main__':
    gen = Data_Generator('../../data/pupil_dataset/data.h5',
                         batch_size=2,
                         data_type=Generator.DATA_TYPE_TOTAL_TEST,
                         shuffle=False)

    for i in range(gen.__len__()):
        xs, ys = gen.__getitem__(i)
        n, b, h, w, c = np.shape(xs)
        for j in range(b):
            for k in range(n):
                cv2.imshow('img%d' % k, (xs[k][j] * 255).astype(np.uint8))
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
        if key == ord('q'):
            break
    cv2.destroyAllWindows()