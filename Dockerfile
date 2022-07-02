FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
RUN pip install -e detectron2_repo

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y


RUN pip install pandas
COPY ./app /app
