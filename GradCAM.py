import io
import keras
import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("cxr_normal_tb_vgg16_model.keras")


class Grad_CAM_class():
    def __init__(self, model, last_conv_layer_name, img_bytes, classes_names, size):
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name
        self.img_bytes = img_bytes
        self.classes_names = classes_names
        self.size = size


    def decode_predictions(self, preds):
        return self.classes_names[round(preds[0][0])]
    
    def get_img_array(self): 
        img = io.BytesIO(self.img_bytes)
        array = keras.utils.img_to_array(img)
        array = np.expand_dims(array, axis=0)
        return array
    
    def make_gradcam_heatmap(self):
        grad_model = keras.models.Model(
            self.model.inputs, [self.model.get_layer(self.last_conv_layer_name).output, self.model.output]
        )
        pred_index = None

        with tf.GradientTape as tape:
            self.img = self.get_img_array()
            last_conv_layer_output, preds = grad_model(self.img)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradients(class_channel, last_conv_layer_output)

        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
        last_conv_layer_output =last_conv_layer_output[0]

        self.heatmap = last_conv_layer_output@pooled_grads[..., tf.newaxis]
        self.heatmap = tf.squeeze(self.heatmap)

        self.heatmap = tf.maximum(self.heatmap, 0)/tf.math.reduce_max(self.heatmap)

        return self.heatmap.numpy()
    

    def save_and_display_gracam(self):
        cam_path = "cam.jpg"
        alpha = 0.4
        self.new_heatmap = np.uint8(255*self.heatmap)

        jet = mpl.colormaps["jet"]
        jet_colors = jet(np.arange(256))[:, :3]

        jet_heatmap = jet_colors[self.new_heatmap]
        jet_heatmap = keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize(self.img[1], self.img[0])
        jet_heatmap = keras.utils.img_to_array(jet_heatmap)

        superimposed = jet_heatmap*alpha*self.img

