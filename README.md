**Title** : TB CXR Classification

**Description** : Classifying CXR automatically for TB can help clinics and hospitals triage patients with patients with abnormal CXR being priotized. This is a web app based CXR classification AI model. It is based on a finetuned VGG16 on tensorflow library and uses streamlit for the web interface GUI. The app also displays the Grad CAM out put of the model to show which pixels the model is using for the classification.



**Getting Started** : For a quick start if you have docker running you can  
- docker run -p 8501:8501 ngwaru/cxr:0.1.7 (on command line)
- Click on then ctrl + click the links provided
- The site will load and then upload your CXR in png or jpeg format
- Click the process image button
- You might need to scroll down to see the results

