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
This project uses a fine-tuned VGG19 convolutional neural network for binary classification of chest X-rays into TB-positive and normal categories. The model is implemented using TensorFlow and trained on a publicly available CXR dataset with appropriate augmentation to enhance generalization.

To enhance trust and interpretability, Grad-CAM (Gradient-weighted Class Activation Mapping) is employed to highlight the regions of the image that most influenced the model's decision. This provides visual feedback to clinicians and users on what part of the lung the AI focused on during classification.

For accessibility and ease of use, the model is wrapped in a Streamlit web application that allows users to upload images, run inference, and view both the prediction and the corresponding Grad-CAM visualization. A prebuilt Docker image is also provided to simplify deployment.
## **Dataset Overview**
The dataset includes both Tuberculosis (TB) positive and Normal chest X-ray images:
- 700 TB-positive images: Publicly accessible in this release.
- 3500 Normal images: Included for comparative analysis.
This dataset is intended to support research in medical imaging, machine learning, and automated TB detection.

## **Training**
This project employs a fine-tuned VGG19 convolutional neural network for binary classification of chest X-ray (CXR) images into TB-positive and Normal categories.
Training Details
- Architecture: VGG19, pre-trained on ImageNet and fine-tuned for medical image classification.
- Framework: Implemented using TensorFlow.
- Dataset: Publicly available chest X-ray dataset containing TB-positive and Normal images.
- Preprocessing:
- Image resizing and normalization
- Training Strategy:
- Transfer learning with frozen base layers initially
- Binary cross-entropy loss with Adam optimizer
Evaluation:
- Accuracy, precision, recall, and AUC metrics used to assess performance
- Validation split applied to monitor overfitting
This setup enables robust classification performance while leveraging the expressive power of deep convolutional features.

## **Results**
<img width="744" height="578" alt="image" src="https://github.com/user-attachments/assets/fe5bf1b8-203d-4630-9a81-138f33ae847a" />


<img width="1003" height="683" alt="Screenshot 2025-09-23 230738" src="https://github.com/user-attachments/assets/29650faa-2214-4a09-82c6-9f026af791f8" />


## **References** 

1. Tawsifur Rahman, Amith Khandakar, Muhammad A. Kadir, Khandaker R. Islam, Khandaker F. Islam, Zaid B. Mahbub, Mohamed Arselene Ayari, Muhammad E. H. Chowdhury. (2020) "Reliable Tuberculosis Detection using Chest X-ray with Deep Learning, Segmentation and Visualization". IEEE Access, Vol. 8, pp 191586 - 191601. DOI. 10.1109/ACCESS.2020.3031384
2. Zar Nawab Khan Swati, Qinghua Zhao, Muhammad Kabir, Farman Ali, Zakir Ali, Saeed Ahmed, Jianfeng Lu, Brain tumor classification for MR images using transfer learning and fine-tuning, Computerized Medical Imaging and Graphics, Vol. 75, 2019, pp 34-46, ISSN 0895-6111, https://doi.org/10.1016/j.compmedimag.2019.05.001. (https://www.sciencedirect.com/science/article/pii/S0895611118305937)
