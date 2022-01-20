mount-files:
	docker run -it -v $(PWD)/:/app/ fydp_sandbox:latest /bin/bash