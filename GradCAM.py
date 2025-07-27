import io
import keras
import numpy as np
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

preprocess_input = keras.applications.xception.preprocess_input

class Grad_CAM_class():
    def __init__(self, model, last_conv_layer_name, img_path, classes_names, size):
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name
        # self.img_not_changed = keras.utils.array_to_img(img)
        self.img_path = img_path
        self.classes_names = classes_names
        self.size = size
        self.img_array = None


    def decode_predictions(self, preds):
        return self.classes_names[round(preds[0][0])]
    

    def get_img_array(self): 
        self.img = keras.utils.load_img(self.img_path, target_size=self.size)
        self.img_array = keras.utils.img_to_array(self.img)
        self.img_array_expanded  = np.expand_dims(self.img_array, axis=0)
        return self.img_array 
       
    def make_gradcam_heatmap(self):
        grad_model = keras.models.Model(
            self.model.inputs, [self.model.get_layer(self.last_conv_layer_name).output, self.model.output]
        )
        pred_index = None

        with tf.GradientTape() as tape:
            # self.img = get_img_array(self)
            last_conv_layer_output, preds = grad_model(preprocess_input(self.img_array_expanded))
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)

        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
        last_conv_layer_output = last_conv_layer_output[0]

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
        jet_heatmap = jet_heatmap.resize((self.img_array.shape[1], self.img_array.shape[0]), resample=Image.Resampling.LANCZOS)
        jet_heatmap = keras.utils.img_to_array(jet_heatmap)

        superimposed_img = jet_heatmap*alpha*self.img
        superimposed_img = keras.utils.array_to_img(superimposed_img)
        superimposed_img.save(cam_path)
        new_image = Image.open(cam_path)

        return new_image

