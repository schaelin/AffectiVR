from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *


def resnet_network(input_shape=(240, 320, 1), name='backbone', is_heatmap=False):
    x = Input(shape=input_shape)

    c1 = Conv2D(32, 7, strides=2, padding='same')(x)
    c1 = BatchNormalization()(c1)
    c1 = Activation(activation='relu')(c1)
    c1 = MaxPooling2D(3, strides=2, padding='same')(c1)

    def resnet_block(x, filters, kernel_size, first_strides, shortcut_conv):
        # Short cut
        s = x
        if shortcut_conv:
            s = Conv2D(filters, kernel_size, strides=first_strides, padding='same')(s)
            s = BatchNormalization()(s)
        # Convolution
        c = Conv2D(filters, kernel_size, strides=first_strides, padding='same')(x)
        c = BatchNormalization()(c)
        c = Activation(activation='relu')(c)
        c = Conv2D(filters, kernel_size, strides=1, padding='same')(c)
        c = BatchNormalization()(c)
        # Merge
        c = Add()([c, s])
        return c

    c2 = resnet_block(c1, 32, 3, 1, False)
    c2 = Activation(activation='relu')(c2)
    c2 = resnet_block(c2, 32, 3, 1, False)
    if is_heatmap:
        m1 = Model(x, c2)
        x = Input(K.int_shape(c2)[1:])
        c2 = x
    c2 = Activation(activation='relu')(c2)
    f1 = GlobalAveragePooling2D()(c2)

    c3 = resnet_block(c2, 64, 3, 2, True)
    c3 = Activation(activation='relu')(c3)
    c3 = resnet_block(c3, 64, 3, 1, False)
    if is_heatmap:
        m2 = Model(x, c3)
        x = Input(K.int_shape(c3)[1:])
        c3 = x
    c3 = Activation(activation='relu')(c3)
    f2 = GlobalAveragePooling2D()(c3)

    c4 = resnet_block(c3, 128, 3, 2, True)
    c4 = Activation(activation='relu')(c4)
    c4 = resnet_block(c4, 128, 3, 1, False)
    if is_heatmap:
        m3 = Model(x, c4)
        x = Input(K.int_shape(c4)[1:])
        c4 = x
    c4 = Activation(activation='relu')(c4)
    f3 = GlobalAveragePooling2D()(c4)

    c5 = resnet_block(c4, 256, 3, 2, True)
    c5 = Activation(activation='relu')(c5)
    c5 = resnet_block(c5, 256, 3, 1, False)
    if is_heatmap:
        m4 = Model(x, c5)
        x = Input(K.int_shape(c5)[1:])
        c5 = x
    c5 = Activation(activation='relu')(c5)
    f4 = GlobalAveragePooling2D()(c5)

    if is_heatmap:
        f1 = Input(K.int_shape(f1)[1:], name='v1')
        f2 = Input(K.int_shape(f2)[1:], name='v2')
        f3 = Input(K.int_shape(f3)[1:], name='v3')
        f4 = Input(K.int_shape(f4)[1:], name='v4')

    v1 = Dense(32, use_bias=False)(f1)
    v2 = Dense(64, use_bias=False)(f2)
    v3 = Dense(128, use_bias=False)(f3)
    v4 = Dense(256, use_bias=False)(f4)

    v = Concatenate()([v1, v2, v3, v4])
    if is_heatmap:
        m5 = Model([f1, f2, f3, f4], v)
        return [m1, m2, m3, m4, m5]
    else:
        model = Model(x, v, name=name)
        return model


def resnet34_network(input_shape=(240, 320, 1), name='backbone', is_heatmap=False):
    x = Input(shape=input_shape)

    c1 = Conv2D(32, 7, strides=2, padding='same')(x)
    c1 = BatchNormalization()(c1)
    c1 = Activation(activation='relu')(c1)
    c1 = MaxPooling2D(3, strides=2, padding='same')(c1)

    def resnet_block(x, filters, kernel_size, first_strides, shortcut_conv):
        # Short cut
        s = x
        if shortcut_conv:
            s = Conv2D(filters, kernel_size, strides=first_strides, padding='same')(s)
            s = BatchNormalization()(s)
        # Convolution
        c = Conv2D(filters, kernel_size, strides=first_strides, padding='same')(x)
        c = BatchNormalization()(c)
        c = Activation(activation='relu')(c)
        c = Conv2D(filters, kernel_size, strides=1, padding='same')(c)
        c = BatchNormalization()(c)
        # Merge
        c = Add()([c, s])
        return c

    c2 = resnet_block(c1, 32, 3, 1, False)
    c2 = Activation(activation='relu')(c2)
    c2 = resnet_block(c1, 32, 3, 1, False)
    c2 = Activation(activation='relu')(c2)
    c2 = resnet_block(c2, 32, 3, 1, False)
    if is_heatmap:
        m1 = Model(x, c2)
        x = Input(K.int_shape(c2)[1:])
        c2 = x
    c2 = Activation(activation='relu')(c2)
    f1 = GlobalAveragePooling2D()(c2)

    c3 = resnet_block(c2, 64, 3, 2, True)
    c3 = Activation(activation='relu')(c3)
    c3 = resnet_block(c3, 64, 3, 1, False)
    c3 = Activation(activation='relu')(c3)
    c3 = resnet_block(c3, 64, 3, 1, False)
    c3 = Activation(activation='relu')(c3)
    c3 = resnet_block(c3, 64, 3, 1, False)
    if is_heatmap:
        m2 = Model(x, c3)
        x = Input(K.int_shape(c3)[1:])
        c3 = x
    c3 = Activation(activation='relu')(c3)
    f2 = GlobalAveragePooling2D()(c3)

    c4 = resnet_block(c3, 128, 3, 2, True)
    c4 = Activation(activation='relu')(c4)
    c4 = resnet_block(c4, 128, 3, 1, False)
    c4 = Activation(activation='relu')(c4)
    c4 = resnet_block(c4, 128, 3, 1, False)
    c4 = Activation(activation='relu')(c4)
    c4 = resnet_block(c4, 128, 3, 1, False)
    c4 = Activation(activation='relu')(c4)
    c4 = resnet_block(c4, 128, 3, 1, False)
    c4 = Activation(activation='relu')(c4)
    c4 = resnet_block(c4, 128, 3, 1, False)
    if is_heatmap:
        m3 = Model(x, c4)
        x = Input(K.int_shape(c4)[1:])
        c4 = x
    c4 = Activation(activation='relu')(c4)
    f3 = GlobalAveragePooling2D()(c4)

    c5 = resnet_block(c4, 256, 3, 2, True)
    c5 = Activation(activation='relu')(c5)
    c5 = resnet_block(c5, 256, 3, 1, False)
    c5 = Activation(activation='relu')(c5)
    c5 = resnet_block(c5, 256, 3, 1, False)
    if is_heatmap:
        m4 = Model(x, c5)
        x = Input(K.int_shape(c5)[1:])
        c5 = x
    c5 = Activation(activation='relu')(c5)
    f4 = GlobalAveragePooling2D()(c5)

    if is_heatmap:
        f1 = Input(K.int_shape(f1)[1:], name='v1')
        f2 = Input(K.int_shape(f2)[1:], name='v2')
        f3 = Input(K.int_shape(f3)[1:], name='v3')
        f4 = Input(K.int_shape(f4)[1:], name='v4')

    v1 = Dense(32, use_bias=False)(f1)
    v2 = Dense(64, use_bias=False)(f2)
    v3 = Dense(128, use_bias=False)(f3)
    v4 = Dense(256, use_bias=False)(f4)

    v = Concatenate()([v1, v2, v3, v4])
    if is_heatmap:
        m5 = Model([f1, f2, f3, f4], v)
        return [m1, m2, m3, m4, m5]
    else:
        model = Model(x, v, name=name)
        return model


def senet(input_shape=(240, 320, 1), name='backbone', is_heatmap=False):
    x = Input(shape=input_shape)

    c1 = Conv2D(32, 7, strides=2, padding='same')(x)
    c1 = BatchNormalization()(c1)
    c1 = Activation(activation='relu')(c1)
    c1 = MaxPooling2D(3, strides=2, padding='same')(c1)

    def resnet_block(x, filters, kernel_size, first_strides, shortcut_conv):
        # Short cut
        s = x
        if shortcut_conv:
            s = Conv2D(filters, 1, strides=2, padding='same')(s)
            s = BatchNormalization()(s)
        # Convolution
        c = Conv2D(filters, kernel_size, strides=first_strides, padding='same')(x)
        c = BatchNormalization()(c)
        c = Activation(activation='relu')(c)
        c = Conv2D(filters, kernel_size, strides=1, padding='same')(c)
        c = BatchNormalization()(c)
        # Merge
        c = Add()([c, s])
        return c

    def se_resnet_block(x, filters, kernel_size, first_strides, shortcut_conv):
        # Short cut
        s = x
        if shortcut_conv:
            s = Conv2D(filters, kernel_size, strides=first_strides, padding='same')(s)
            s = BatchNormalization()(s)
        # Convolution
        c = Conv2D(filters, kernel_size, strides=first_strides, padding='same')(x)
        c = BatchNormalization()(c)
        c = Activation(activation='relu')(c)
        c = Conv2D(filters, kernel_size, strides=1, padding='same')(c)
        c = BatchNormalization()(c)
        c = se_block(c)
        # Merge
        x = Add()([c, s])
        return x

    def se_block(x):
      ratio = 16
      n_channel = int(x.shape[-1])
      x1 = x
      x1 = GlobalAveragePooling2D()(x1)
      x1 = Reshape((1,1,n_channel))(x1)
      x1 = Dense(n_channel//ratio)(x1)
      x1 = Activation(activation='relu')(x1)
      x1 = Dense(n_channel)(x1)
      x1 = Activation(activation='sigmoid')(x1)

      return Multiply()([x,x1])

    c2 = resnet_block(c1, 32, 3, 1, False)
    c2 = Activation(activation='relu')(c2)
    c2 = resnet_block(c2, 32, 3, 1, False)
    if is_heatmap:
        m1 = Model(x, c2)
        x = Input(K.int_shape(c2)[1:])
        c2 = x
    c2 = Activation(activation='relu')(c2)
    f1 = GlobalAveragePooling2D()(c2)

    c3 = resnet_block(c2, 64, 3, 2, True)
    c3 = Activation(activation='relu')(c3)
    c3 = resnet_block(c3, 64, 3, 1, False)
    if is_heatmap:
        m2 = Model(x, c3)
        x = Input(K.int_shape(c3)[1:])
        c3 = x
    c3 = Activation(activation='relu')(c3)
    f2 = GlobalAveragePooling2D()(c3)

    c4 = resnet_block(c3, 128, 3, 2, True)
    c4 = Activation(activation='relu')(c4)
    c4 = se_resnet_block(c4, 128, 3, 1, True)
    if is_heatmap:
        m3 = Model(x, c4)
        x = Input(K.int_shape(c4)[1:])
        c4 = x
    c4 = Activation(activation='relu')(c4)
    f3 = GlobalAveragePooling2D()(c4)

    c5 = resnet_block(c4, 256, 3, 2, True)
    c5 = Activation(activation='relu')(c5)
    c5 = se_resnet_block(c5, 256, 3, 1, True)
    if is_heatmap:
        m4 = Model(x, c5)
        x = Input(K.int_shape(c5)[1:])
        c5 = x
    c5 = Activation(activation='relu')(c5)
    f4 = GlobalAveragePooling2D()(c5)

    if is_heatmap:
        f1 = Input(K.int_shape(f1)[1:], name='v1')
        f2 = Input(K.int_shape(f2)[1:], name='v2')
        f3 = Input(K.int_shape(f3)[1:], name='v3')
        f4 = Input(K.int_shape(f4)[1:], name='v4')

    v1 = Dense(32, use_bias=False)(f1)
    v2 = Dense(64, use_bias=False)(f2)
    v3 = Dense(128, use_bias=False)(f3)
    v4 = Dense(256, use_bias=False)(f4)

    v = Concatenate()([v1, v2, v3, v4])
    if is_heatmap:
        m5 = Model([f1, f2, f3, f4], v)
        return [m1, m2, m3, m4, m5]
    else:
        model = Model(x, v, name=name)
        return model


def get_network(input_shape, feature_type, name='backbone', is_heatmap=False):
    if feature_type == 'deep_resnet':
        model = resnet_network(input_shape=input_shape, name=name, is_heatmap=is_heatmap)
    elif feature_type == 'resnet34':
        model = resnet34_network(input_shape=input_shape, name=name, is_heatmap=is_heatmap)
    elif feature_type == 'senet':
        model = senet(input_shape=input_shape, name=name, is_heatmap=is_heatmap)
    else:
        model = None
    return model


def train_model(input_shape=(240, 320, 1), feature_type='cnn'):
    model = get_network(input_shape, feature_type)

    input1 = Input(input_shape)
    input2 = Input(input_shape)

    v1 = model(input1)
    v2 = model(input2)

    a = 10
    b = 5

    sub = Subtract()([v1, v2])

    d = Lambda(lambda x: K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True)))(sub)
    s = Lambda(lambda x: b - a * K.square(x))(d)
    p = Activation('sigmoid')(s)
    m = Model([input1, input2], p)
    return m


def test_model(input_shape=(240, 320, 1), feature_type='cnn'):
    model = get_network(input_shape, feature_type)

    input1 = Input(input_shape, name='input1')
    input2 = Input(input_shape, name='input2')

    vector1 = model(input1)
    vector2 = model(input2)

    d = Subtract()([vector1, vector2])
    d = Lambda(lambda x: K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True)))(d)

    m = Model([input1, input2], d)
    return m


def test_model_for_heatmap(input_shape=(240, 320, 1), feature_type='cnn'):
    models = get_network(input_shape, feature_type, is_heatmap=True)

    input1 = Input(input_shape, name='input1')
    input2 = Input(input_shape, name='input2')

    x1 = input1
    x2 = input2
    v1, v2 = [], []
    pool = GlobalAveragePooling2D()

    for i, m in enumerate(models[:-1]):
        x1 = m(x1)
        x2 = m(x2)

        x1 = Lambda(lambda x: x, name='for_gradient_1_%d' % (i + 1))(x1)
        x2 = Lambda(lambda x: x, name='for_gradient_2_%d' % (i + 1))(x2)

        x1_v = Lambda(lambda x: x, name='for_vector_1_%d' % (i + 1))(x1)
        x2_v = Lambda(lambda x: x, name='for_vector_2_%d' % (i + 1))(x2)

        x1 = Lambda(lambda x: x, name='for_network_1_%d' % (i + 1))(x1)
        x2 = Lambda(lambda x: x, name='for_network_2_%d' % (i + 1))(x2)

        v1.append(pool(x1_v))
        v2.append(pool(x2_v))

    vector1 = models[-1](v1)
    vector2 = models[-1](v2)

    d = Subtract()([vector1, vector2])
    d = Lambda(lambda x: K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True)))(d)

    m = Model([input1, input2], d)
    return m


if __name__ == '__main__':
    input_shape = (240, 320, 1)
    feature_type = 'deep_resnet'
    model = train_model(input_shape, feature_type)
    model.summary()
    model.get_layer('backbone').summary()