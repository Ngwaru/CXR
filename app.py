from taipy.gui import Gui
import numpy as np
from tensorflow.keras import models
from PIL import Image

content = ""
img_path = "placeholder.jpg"

prob = 0
display_message = ""

model = models.load_model("cxr_normal_tb_vgg16_model.keras")

def predict_image(model, path_to_image):
    image = Image.open(path_to_image)
    image = image.convert("RGB")
    image = image.resize((200, 200))
    data = np.asarray(image)
    data = data/255
    prob_of_tb = model.predict(np.array([data]))
   
    if prob_of_tb > 0.5:
        message = "TB is likely"
    elif prob_of_tb > 0.4:
        message = "TB cannot be ruled out"
    else:
        message = "TB is unlikely"
    return prob_of_tb, message


index = """
<|text-center|
<|{"logo.png"}|image|>

Upload a CXR to review

<|{content}|file_selector|extensions= .png,.jpg|>



<|{display_message}|>

<|{img_path}|image|>

<|{prob}|indicator|value={prob}|min=0|max=100|width=40vw|>
>
"""

def on_change(state, var_name, var_val):
    if var_name == "content":
        state.img_path = var_val
        prob_of_tb, message = predict_image(model, var_val)
        print(prob_of_tb)
        state.prob = prob_of_tb[0][0]*100

        state.display_message = message


app = Gui(page=index)

if  __name__ == "__main__":
    app.run(use_reloader=True)


# Next Steps
# 1. Train on more data
# 2. Augument the Data
# 3. Create a model that identifies if an image is a CXR before predicting if TB is present or not 
# 4. Send an Alert is TB is likely
