import numpy as np
import glob
import cv2
import h5py
import random


# get image file list
def get_file_list(directory_path):
    cut_num = 105  # 몇 장씩 자를 건지 나누는 변수
    paths = glob.glob(directory_path)
    paths_cut = []

    for path in paths:
        tmp = glob.glob(path + '/*.png')

        # 인당 random 추출 후 경로 리스트에 저장
        for i in range(cut_num):
            idx = random.randrange(0, len(tmp))
            paths_cut.append(tmp.pop(idx))

    print(len(set(paths_cut)))

    return paths_cut


def get_image_data(paths, channels=1, log=True, log_skip=1000):
    if channels == 1:
        image_type = cv2.CV_8UC1
    elif channels == 3:
        image_type = cv2.CV_8UC3
    else:
        print('Undefined image channels.')
        return False

    images = []
    for i, path in enumerate(paths):
        if log and i % log_skip == 0:
            print('%05dth data' % i)
        image = cv2.imread(path, image_type)
        images.append(image)
    return np.array(images)


def resize_datas(frames, size=(320, 240), interpolation=cv2.INTER_LINEAR):
    new_frames = []
    for frame in frames:
        new_frame = cv2.resize(frame, dsize=size, interpolation=interpolation)
        new_frames.append(new_frame)
    return np.array(new_frames, np.uint8)


def get_index_and_count(paths):
    names = set()
    flag = True
    for path in paths:
        # 경로마다 25:29 index 범위 수정 필요
        name = path[25:29]

        # idx 범위 파악 위해 첫 이미지 경로만 출력
        if flag:
            print(name)
            flag = False

        names.add(name)
    names = sorted(names)

    index, count = [], []
    total = 0

    for name in names:
        # count 위해 경로에 들어있는 회수 세기
        c = sum(name in s for s in paths)

        index.append(total)
        count.append(c)

        total += c

    return index, count


# 6:2:2 dataset split
def split_index_and_count_to_train_validation_test(index, count, train_ratio=0.6, validation_ratio=0.2):
    merge_list = list(zip(index, count))
    random.shuffle(merge_list)
    temp_index, temp_count = zip(*merge_list)

    train_end = int(len(index) * train_ratio)
    validation_start = train_end
    validation_end = validation_start + int(len(index) * validation_ratio)
    test_start = validation_end

    train_index = temp_index[: train_end]
    validation_index = temp_index[validation_start: validation_end]
    test_index = temp_index[test_start:]

    train_count = temp_count[: train_end]
    validation_count = temp_count[validation_start: validation_end]
    test_count = temp_count[test_start:]

    return (train_index, train_count), (validation_index, validation_count), (test_index, test_count)


def create_dataset(path, dest_path):
    paths = get_file_list(path)
    images = get_image_data(paths, channels=1, log=True, log_skip=1000)

    images = resize_datas(images)
    images = np.expand_dims(images, axis=-1)

    total = get_index_and_count(paths)

    total_train, total_validation, total_test = split_index_and_count_to_train_validation_test(total[0], total[1])

    data_file = h5py.File(dest_path, mode='w')

    data_file.create_dataset('images', images.shape, np.uint8)
    data_file.create_dataset('index', np.shape(total[0]), np.uint16)
    data_file.create_dataset('train_index', np.shape(total_train[0]), np.uint16)
    data_file.create_dataset('validation_index', np.shape(total_validation[0]), np.uint16)
    data_file.create_dataset('test_index', np.shape(total_test[0]), np.uint16)

    data_file.create_dataset('count', np.shape(total[1]), np.uint16)
    data_file.create_dataset('train_count', np.shape(total_train[1]), np.uint16)
    data_file.create_dataset('validation_count', np.shape(total_validation[1]), np.uint16)
    data_file.create_dataset('test_count', np.shape(total_test[1]), np.uint16)

    data_file['images'][...] = images

    data_file['index'][...] = total[0]
    data_file['train_index'][...] = total_train[0]
    data_file['validation_index'][...] = total_validation[0]
    data_file['test_index'][...] = total_test[0]

    data_file['count'][...] = total[1]
    data_file['train_count'][...] = total_train[1]
    data_file['validation_count'][...] = total_validation[1]
    data_file['test_count'][...] = total_test[1]

    data_file.close()


def check_dataset(path):
    file = h5py.File(path, 'r')

    frames = file['images']
    index = file['index']
    count = file['count']

    print(index[:])
    print(count[:])

    for idx, c in zip(index, count):
        for i in range(c):
            frame = frames[idx + i]
            cv2.imshow('frame', frame)
            key = cv2.waitKey(0)
            if key == ord('q'):
                return
            elif key == ord('n'):
                return
            print(idx, i)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    """
    /P*/, */*.png : 인당 N장
    /P*/*, /*.png : 영상당 N장
    """
    # name idx 범위 찾는 용도
    paths = get_file_list('../../data/pupil_dataset/P*/')
    total, left = get_index_and_count(paths)
    print(total)

    create_dataset('../../data/pupil_dataset/P*/', '../../data/pupil_dataset/data.h5')

    files = ['../../data/pupil_dataset/data.h5']
    for f in files:
        check_dataset(f)
