IMAGE_NAME:=fydp-sandbox
IMAGE_VERSION:=latest
mount-files:
	docker run -it  -v $(PWD)/:/app/ $(IMAGE_NAME):$(IMAGE_VERSION) /bin/bash

build-image:
	docker build . -t $(IMAGE_NAME):$(IMAGE_VERSION)

jupyter-server:
	docker run -it -p 8888:8888 -v $(PWD)/:/app/ $(IMAGE_NAME):$(IMAGE_VERSION)