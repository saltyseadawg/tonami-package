IMAGE_NAME:=fydp_sandbox
IMAGE_VERSION:=latest
mount-files:
	docker run -it -v $(PWD)/:/app/ $(IMAGE_NAME):$(IMAGE_VERSION) /bin/bash

build-image:
	docker build . -t $(IMAGE_NAME):$(IMAGE_VERSION)