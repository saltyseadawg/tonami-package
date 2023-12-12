.PHONY: mount-files build-image push-image pull-image jupyter-server heroku-server lint build-app

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

heroku-server:
	docker run -it -p 8501:8501 -v $(PWD)/:/app/ saltyseadawg/tonami-app:$(IMAGE_VERSION) /bin/bash -c \
	"streamlit run tonami_interface.py"

# code formatting
# TODO: exclude jupyter notebook checkpoints, as they are auto saved versions of the file
lint:
	docker run -it  -v $(PWD)/:/app/ $(IMAGE_NAME):$(IMAGE_VERSION) /bin/bash -c \
	"black tonami/ tests/ dev/ data_viz/; \
	flake8 tonami/ tests/ dev/ data_viz/"

# run inside container
test:
	python -m pytest tests/

# run outside of container
test-container:
	docker run -it  -v $(PWD)/:/app/ $(IMAGE_NAME):$(IMAGE_VERSION) /bin/bash -c \
	"python -m pytest tests/"

# release commands
build-app:
	docker build -f Dockerfile.app -t saltyseadawg/tonami-app:latest .

push-app:
	docker push saltyseadawg/tonami-app:latest

image-id:
	docker inspect --format="{{.Id}}" registry.heroku.com/tonami-testing/web


