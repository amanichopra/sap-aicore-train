venv:
	python3 -m venv .venv

setup-env:
	pip3 install -r app/requirements.txt

cf-login-sap:
	cf login -a https://api.cf.sap.hana.ondemand.com -o SLS-ATI-ML_testd070430 -s second --sso

push-docker-image:
	docker build . -t amanichopra/aicore-train:latest --platform linux/amd64
	docker push amanichopra/aicore-train:latest

deploy-container-service-local:
	docker build . -t amanichopra/aicore-train:local --platform linux/amd64
	docker run -it amanichopra/aicore-train:local


export-env:
	export $(grep -v '^#' .env | xargs)

run-app-local:
	python3 app/main.py 0 8090