
import tensorflow as tf
from keras.optimizers import Adam
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense

# 确认是否使用 GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("TensorFlow version:", tf.__version__)
print("Available devices:")

# 建立預訓練模型
def build_model(num_classes=7, input_shape=(48, 48, 3)):
    def inverted_res_block(x, filters, alpha, stride, expansion):
        in_channels = int(x.shape[-1])
        pointwise_conv_filters = int(filters * alpha)
        # Expansion Layer
        x = Conv2D(expansion * in_channels, kernel_size=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        # 深度可分離卷積（Depthwise Separable Convolution）
        x = DepthwiseConv2D(kernel_size=(3, 3), strides=stride, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        # Pointwise Convolution
        x = Conv2D(pointwise_conv_filters, kernel_size=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        # Linear function replace ReLU
        x = Activation('linear')(x)
        # Residual Connection
        if in_channels == pointwise_conv_filters and stride == 1:
            x = Add()([x, inputs])
        return x
    inputs = Input(shape=input_shape)
    x = inverted_res_block(inputs, filters=16, alpha=1.0, stride=1, expansion=1)
    x = inverted_res_block(x, filters=24, alpha=1.0, stride=2, expansion=6)
    x = inverted_res_block(x, filters=32, alpha=1.0, stride=2, expansion=6)
    x = inverted_res_block(x, filters=64, alpha=1.0, stride=2, expansion=6)
    x = inverted_res_block(x, filters=96, alpha=1.0, stride=1, expansion=6)
    x = inverted_res_block(x, filters=160, alpha=1.0, stride=2, expansion=6)
    x = inverted_res_block(x, filters=320, alpha=1.0, stride=1, expansion=6)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)# add dropout
    x = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, x, name='mobilenetv2')
    optimizer = Adam(lr=1e-3)  # 調整學習率
    # 編譯模型
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# train(build_model(),"MobilenetV2")

