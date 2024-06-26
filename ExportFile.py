import tensorflow as tf
from tensorflowimpl import MobilenetV2Implement
from tensorflowimpl.ModelTraining import train


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

# 產生mobilenet_v2.tflite
model_name = "mobilenet_v2"
save_model(train(MobilenetV2Implement.build_model()), model_name)

# tflite_support會有版本問題,可以用colab生成 model metadata
# ImageClassifierWriter = image_classifier.MetadataWriter
# _MODEL_PATH = "./exportfile/mobilenet_v2.tflite"
# _LABEL_FILE = "./exportfile/labels.txt"
# _SAVE_TO_PATH = "./exportfile/effB0_fer_meta.tflite"
#
# _INPUT_NORM_MEAN = 0
# _INPUT_NORM_STD = 1
#
# # Create the metadata writer.
# writer = ImageClassifierWriter.create_for_inference(
#     writer_utils.load_file(_MODEL_PATH), [_INPUT_NORM_MEAN], [_INPUT_NORM_STD],
#     [_LABEL_FILE])
# # Verify the metadata generated by metadata writer.
# print(writer.get_metadata_json())
# # Populate the metadata into the model.
# writer_utils.save_file(writer.populate(), _SAVE_TO_PATH)


