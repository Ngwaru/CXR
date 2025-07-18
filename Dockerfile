FROM python:3.11
ADD app.py .
ADD cxr_normal_tb_vgg16_model.keras .
ADD cxr_or_not_vgg16_model.keras .
ADD logo.png .
ADD placeholder.jpg .
ADD requirements.txt .
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
