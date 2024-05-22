from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense


# 建立預訓練模型
def build_model(num_classes=7, input_shape=(48, 48, 1)):
    def depthwise_separable_conv(x, filters, alpha, stride):
        # Depthwise Convolution
        x = DepthwiseConv2D(kernel_size=(3, 3), strides=stride, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        # Pointwise Convolution
        pointwise_conv_filters = int(filters * alpha)
        x = Conv2D(pointwise_conv_filters, kernel_size=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        return x

    # Convolution Layer
    inputs = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = depthwise_separable_conv(x, filters=64, alpha=1.0, stride=1)
    x = depthwise_separable_conv(x, filters=128, alpha=1.0, stride=2)
    x = depthwise_separable_conv(x, filters=128, alpha=1.0, stride=1)
    x = depthwise_separable_conv(x, filters=256, alpha=1.0, stride=2)
    x = depthwise_separable_conv(x, filters=256, alpha=1.0, stride=1)
    x = depthwise_separable_conv(x, filters=512, alpha=1.0, stride=2)

    # Pooling Layer
    x = GlobalAveragePooling2D()(x)
    # Fully Connected Layer
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, x, name='mobilenetv1')

    # 編譯模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# train(build_model(),"Mobilenet")