FROM python:3.11
# ADD app.py .
ADD main.py .
ADD GradCAM.py .
ADD cxr_normal_tb_vgg19_324_model.keras .
ADD cxr_or_not_vgg16_model.keras .
ADD logo.png .
ADD placeholder.jpg .
ADD requirements.txt .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "main.py", "--server.fileWatcherType", "none"]
