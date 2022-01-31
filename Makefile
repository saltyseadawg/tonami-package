.PHONY: mount-files build-image push-image pull-image jupyter-server lint

IMAGE_NAME:=saltyseadawg/tonami-package
IMAGE_VERSION:=latest

# Open cmd line in docker container and mount working directory
mount-files:
	docker run -it  -v $(PWD)/:/app/ $(IMAGE_NAME):$(IMAGE_VERSION) /bin/bash

build-image:
	docker build . -t $(IMAGE_NAME):$(IMAGE_VERSION)

push-image:
	docker push $(IMAGE_NAME):$(IMAGE_VERSION)

pull-image:
	docker pull $(IMAGE_NAME):$(IMAGE_VERSION)

jupyter-server:
	docker run -it -p 8888:8888 -v $(PWD)/:/app/ $(IMAGE_NAME):$(IMAGE_VERSION)

# code formatting
lint:
	docker run -it  -v $(PWD)/:/app/ $(IMAGE_NAME):$(IMAGE_VERSION) /bin/bash -c \
	"black tonami/ tests/; \
	flake8 tonami/ tests/"
