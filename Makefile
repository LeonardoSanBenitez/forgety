.ONESHELL: # Source: https://stackoverflow.com/a/30590240

auth:
	echo "Login not yet configured"

run-local: auth
	echo "go to http://localhost:8501"
	if docker compose ps | grep platform-frontend >/dev/null; then \
		echo "Container is already running."; \
	else \
		echo "Container is not running. Starting it now..."; \
		docker compose down; \
		docker compose up -d; \
	fi;

stop-local:
	docker compose down

test: run-local
	echo "\n\n-------\nMypy checks (libs)\n-------"
	docker compose exec backend mypy //src/libs --no-warn-incomplete-stub --disable-error-code import-untyped --explicit-package-bases --install-types --non-interactive --exclude vision_unlearning_benchmarks_I_care_TEMP

	echo "\n\n-------\nMypy checks (backend)\n-------"
	docker compose exec backend mypy //src/app --no-warn-incomplete-stub --disable-error-code import-untyped --explicit-package-bases --install-types --non-interactive

	echo "\n\n-------\nPycodestyle checks (libs)\n-------"
	docker compose exec backend pycodestyle --exclude='.venv,docs,.runs,vision_unlearning_benchmarks_I_care_TEMP.py' --max-line-length=200 --ignore='E121,E123,E126,E226,E24,E251,E704,W503,W504,E225,E226,E252,W605,E721,E731' //src/libs

	echo "\n\n-------\nPycodestyle checks (backend)\n-------"
	docker compose exec backend pycodestyle --exclude='.venv,docs,.runs' --max-line-length=200 --ignore='E121,E123,E126,E226,E24,E251,E704,W503,W504,E225,E226,E252,W605,E721,E731' //src/app

	echo "\n\n-------\nPytest checks\n-------"
	docker compose exec backend python3 -m pytest //tests
