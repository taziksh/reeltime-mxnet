# reeltime-mxnet

## Get started

```
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 829102044026.dkr.ecr.us-east-1.amazonaws.com
docker build --platform=linux/amd64 -t reeltime-mxnet .
docker tag reeltime-mxnet:latest 829102044026.dkr.ecr.us-east-1.amazonaws.com/reeltime-mxnet:latest
docker push 829102044026.dkr.ecr.us-east-1.amazonaws.com/reeltime-mxnet:latest
```
