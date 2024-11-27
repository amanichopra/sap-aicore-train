venv:
	python3 -m venv .venv

setup-env:
	pip3 install -r app/requirements.txt

push-docker-image:
	docker build . -t amanichopra/aicore-train:latest --platform linux/amd64
	docker push amanichopra/aicore-train:latest

deploy-container-service-local:
	docker build . -t amanichopra/aicore-train:local --platform linux/amd64
	docker rm local-aicore
	docker run -it --entrypoint /bin/bash --name local-aicore amanichopra/aicore-train:local

copy-data-to-local-container:
	$(eval CONTAINER_ID = $(shell docker ps -aqf "name=local-aicore"))
	docker cp ./data/pose_embeddings.csv ${CONTAINER_ID}:/app/data/pose_embeddings.csv
	docker cp ./data/metadata.csv ${CONTAINER_ID}:/app/data/metadata.csv

cp-s3-data:
	aws s3 cp ./data/metadata.csv s3://hcp-c1a2d095-b523-400a-bf19-94eda5e8d109/data/metadata.csv
	aws s3 cp ./data/pose_embeddings.csv s3://hcp-c1a2d095-b523-400a-bf19-94eda5e8d109/data/pose_embeddings.csv
	aws s3 ls s3://hcp-c1a2d095-b523-400a-bf19-94eda5e8d109/data/