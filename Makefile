install:
	pip install --upgrade pip && pip install -r requirements-dev.txt

lint:
	pylint --diable=R,C app

test:
	pytest -vv --cov-report html --cov=my_project tests/test_*.py

format:
	black *.py