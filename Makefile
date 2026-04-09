.PHONY: setup train serve serve-docker mlflow-ui drift-check tf-init tf-plan tf-apply

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

train:
	PYTHONPATH=. .venv/bin/python -m src.training.train $(ARGS)

serve:
	PYTHONPATH=. MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
	.venv/bin/uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

serve-docker:
	docker-compose up --build

mlflow-ui:
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

drift-check:
	PYTHONPATH=. .venv/bin/python -m src.monitoring.drift

tf-init:
	cd infra/terraform && terraform init

tf-plan:
	cd infra/terraform && terraform plan

tf-apply:
	cd infra/terraform && terraform apply -auto-approve

tf-destroy:
	cd infra/terraform && terraform destroy -auto-approve
