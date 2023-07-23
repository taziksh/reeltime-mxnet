FROM public.ecr.aws/lambda/python:3.7

COPY lambda_handler.py ./
COPY requirements.txt ./

RUN yum install -y gcc-c++ pkgconfig poppler-cpp-devel libquadmath
RUN python3 -m pip install -r requirements.txt

CMD ["lambda_handler.lambda_handler"]
