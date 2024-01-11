import pandas as pd
import numpy as np
import tensorflow as tf
from IPython.display import display
import mlflow.tensorflow
import MobilenetV2Implement


# 存檔案
def save_model(model, save_name="mymodel"):
    """
    model : Keras Model class
    save_name : str
    """
    # 儲存模型-tf格式
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    # Save the model.
    with open("./exportfile/" + save_name + ".tflite", 'wb') as f:
        f.write(tflite_model)

    # 儲存模型-keras格式
    model.save("./exportfile/" + save_name + ".h5")

# MobilenetV2
model_name = "mobilenet_v2"
save_model(MobilenetV2Implement.build_model(),model_name)
