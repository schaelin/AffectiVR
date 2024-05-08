# Usage

### 1. [Train](src/models/proposed/train_kfold.py)
```python
model_function = train_model
feature_type = 'deep_resnet'
loss = 'binary_crossentropy'

input_shape = (240, 320, 1)
batch_size = 16
epochs = 10

datas = sorted(glob.glob('../../../data/pupil_dataset/data.h5'))

for data in datas:
    print(f'fold {data[-4]} starting...')

    save_path = '../../../result/' + data[-4] + '/model.h5'
    hist_path = '../../../result/' + data[-4] + '/history'

    if not os.path.isdir(save_path[:-9]):
        os.mkdir(save_path[:-9])

    train(model_function, feature_type, loss, data, save_path, hist_path, input_shape, batch_size, epochs)
```

### 2. [Test and save results](src/models/proposed/test.py)
```python
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
```

### 3. [Check results](src/main.py)
```python
EER = []
FRR = []
for i in range(5):
    path = '../result/proposed_resnet.npy'
    EER, FRR = show_test_result(path, True)

    EER.append(round(EER, 2))
    FRR.append(round(FRR, 2))

EER.sort()
FRR.sort()

print(EER, FRR)
print(np.mean(EER), np.mean(FRR))

path = '../result/history'
file = open(path, 'rb')
data = pickle.load(file)

plt.plot(data['loss'])
plt.plot(data['val_loss'])
plt.show()
```
