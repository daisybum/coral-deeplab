# 베이스 이미지: TensorFlow 2.19 GPU
FROM tensorflow/tensorflow:2.12.0-gpu

# 필요 패키지 설치 (git 등)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 작업 디렉터리 생성
WORKDIR /workspace

# (선택) 파이썬 패키지 설치 – 저장소에 필요한 의존성 설치
# 예: requirements.txt가 있을 경우
# COPY requirements.txt ./ 
# RUN pip install -r requirements.txt
# 또는 setup.py를 통해 설치
# RUN pip install .

# (선택) Jupyter/TensorBoard 등을 사용할 경우 포트 노출
# EXPOSE 8888 6006
