## **Title** : Classification of CXR for TB

## **Description** 
Classifying CXR automatically for TB can help clinics and hospitals triage patients.  Patients with abnormal CXR can be priotized. This is a web app based CXR classification AI model. It is based on a finetuned VGG19 usinf a tensorflow library and uses streamlit for the web interface GUI. The app also displays the pixels used for classification using GradCAM.



## **Getting Started**
For a quick start if you have docker running you can  
- docker run -p 8501:8501 ngwaru/cxr:0.1.7 (on the command line)
- ctrl + click on the links provided
- The site will load in the browser and then upload your CXR in png or jpeg format
- Click the process image button
- Scroll down to see the results

## **Motivation**
Tuberculosis (TB) remains one of the leading causes of death from infectious disease globally, especially in low-resource settings. Early detection is critical for effective treatment and to prevent the spread of the disease. Chest X-rays (CXR) are commonly used for TB screening, but interpreting them requires trained radiologists, who may not always be available—particularly in rural or under-resourced areas.

The goal of this project is to develop a lightweight, accessible, and interpretable AI tool that can assist in the preliminary screening of TB from chest X-rays. By automating this step, healthcare providers can prioritize patients with abnormal findings, reduce diagnostic delays, and better allocate medical resources. The integration with a web interface ensures the tool is easy to deploy and use in real-world clinical environments.
## **Approach**

## **Results**


