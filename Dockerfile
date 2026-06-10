FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /workspace

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install torchvision==0.15.1

COPY train.py /workspace/train.py

# 把本地 data 目录复制进镜像
COPY data /workspace/data

# 让 torchvision 在构建镜像时把 raw 数据处理成 processed 数据
# RUN python -c "from torchvision import datasets, transforms; datasets.MNIST('/workspace/data', train=True, download=False, transform=transforms.ToTensor()); datasets.MNIST('/workspace/data', train=False, download=False, transform=transforms.ToTensor())"

ENTRYPOINT ["python", "/workspace/train.py"]