from taipy.gui import Gui
import numpy as np
from tensorflow.keras import models

content = ""
img_path = "placeholder_image.png"


index = """
<|text-center|
<|{"logo.png"}|image|width=40vw|>

<|{content}|file_selector|extensions=.png|>

Upload a CXR to review

<|{img_path}|image|>

<|{label_her}|indicator|value=0|min=0|max=100|width=40vw|>
>
"""

def on_change(state, var_name, var_val):
    if var_name == "content":
        state.img_path = var_val

app = Gui(page=index)

if  __name__ == "__main__":
    app.run(use_reloader=True)

