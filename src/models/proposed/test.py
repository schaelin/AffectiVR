from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

from src.data.generator import Data_Generator
from src.models.proposed.model import test_model
from src.utils.test_util import test

if __name__ == '__main__':
    data_path = '../../../data/pupil_dataset/data.h5'
    batch_size = 128

    path = '../../../result/model_best.h5'
    model = load_model(path, custom_objects={"K": K})
    model.summary()

    data = Data_Generator(data_path, Data_Generator.DATA_TYPE_TOTAL_TEST, batch_size, True)

    feature_type = 'deep_resnet'
    model_path = '../../../result/model_best.h5'
    save_path = '../../../result/proposed_resnet.npy'
    test(model_path=model_path, test_model_func=test_model, gen=data, feature_type=feature_type, save_path=save_path)
