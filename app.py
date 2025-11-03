from tensorflow.keras import models
from taipy.gui import Gui
import numpy as np
from PIL import Image

content = ""
img_path = "placeholder.jpg"

prob = 0
display_message = ""

model = models.load_model("cxr_normal_tb_vgg16_model.keras")

model_xray_or_not = models.load_model("cxr_or_not_vgg16_model.keras")


def is_xray(model, path_to_image):
    image = Image.open(path_to_image)
    image =image.convert("RGB")
    image = image.resize((156,156))
    data = np.asarray(image)
    data = data/255
    xray_or_not = model.predict(np.array([data]))
    print(xray_or_not)
    if xray_or_not > 0.5:
        return True
    else:
        return False




def predict_image(model, path_to_image):
    is_it_an_xray = is_xray(model_xray_or_not, path_to_image)
    if is_it_an_xray:
        image = Image.open(path_to_image)
        image = image.convert("RGB")
        image = image.resize((200, 200))
        data = np.asarray(image)
        data = data/255
        prob_of_tb = model.predict(np.array([data]))
        print(prob_of_tb)
        if prob_of_tb > 0.5:
            message = "TB is likely"
        elif prob_of_tb > 0.4:
            message = "TB cannot be ruled out"
        else:
            message = "TB is unlikely"
        return prob_of_tb, message
    else:
        prob_of_tb = np.array([[0.0]])
        message = "This is not an X-ray"
        return prob_of_tb, message


index = """
<|text-center|
<|{"logo.png"}|image|>

Upload a CXR to review

<|{content}|file_selector|extensions= .png,.jpg,.jpeg|>

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




