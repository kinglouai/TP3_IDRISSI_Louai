UV ?= uv
RUN := $(UV) run --locked
.PHONY: install config report doctor notebook inspect-data check-data train pipeline test mlflow-server server-check snapshot compare validate-submission package clean
install:
	$(UV) sync --locked
config:
	test -f configs/train.yaml || cp configs/train.yaml.example configs/train.yaml
report:
	mkdir -p reports/figures
	test -f reports/tp1_report.md || cp reports/tp1_report_template.md reports/tp1_report.md
doctor:
	@command -v git >/dev/null || { echo "Git est requis"; exit 1; }
	$(RUN) python -c 'import sys, mlflow, sklearn; print(sys.executable); print(sys.version); print("MLflow", mlflow.__version__, "sklearn", sklearn.__version__); assert sys.version_info[:2] == (3, 13)'
	git --version
notebook:
	$(RUN) jupyter notebook
inspect-data:
	$(RUN) python scripts/inspect_data.py
check-data:
	$(RUN) python -m maintenance_ml.modeling.train --check-data
train:
	$(RUN) python -m maintenance_ml.modeling.train
pipeline: train
test:
	$(RUN) python -m pytest -q
mlflow-server:
	$(RUN) mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
server-check:
	@for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do \
		if $(RUN) python -c 'from urllib.request import urlopen; print("MLflow HTTP", urlopen("http://127.0.0.1:5000", timeout=3).status)'; then exit 0; fi; \
		sleep 1; \
	done; \
	echo "MLflow indisponible après 20 s : vérifiez le terminal make mlflow-server"; exit 1
snapshot:
	$(RUN) python scripts/evidence.py snapshot "$(LABEL)"
compare:
	$(RUN) python scripts/evidence.py compare
validate-submission:
	$(RUN) python tests/validate_submission.py
package:
	$(RUN) python scripts/evidence.py package "$(NAME)"
clean:
	$(RUN) python scripts/evidence.py clean
