FROM public.ecr.aws/lambda/python:latest

COPY lambda_handler.py ./
COPY requirements.txt ./

RUN yum install -y gcc-c++ pkgconfig poppler-cpp-devel
RUN python3 -m pip install -r requirements.txt

CMD ["lambda_handler.lambda_handler"]
