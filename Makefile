PYTHON ?= python

.PHONY: test help install-dev clean build check publish publish-test

help:
	@echo "PaPie build / publish targets:"
	@echo "  make test          Run the test suite"
	@echo "  make install-dev   Editable install plus build & twine tooling"
	@echo "  make build         Build sdist + wheel into dist/ (PEP 517)"
	@echo "  make check         Validate the built artifacts with 'twine check'"
	@echo "  make publish-test  Upload dist/* to TestPyPI"
	@echo "  make publish       Upload dist/* to PyPI"
	@echo "  make clean         Remove build artifacts"

test:
	$(PYTHON) -m pytest -q

install-dev:
	$(PYTHON) -m pip install -e . build twine

clean:
	rm -rf dist build *.egg-info

build: clean
	$(PYTHON) -m build

check: build
	$(PYTHON) -m twine check dist/*

publish-test: check
	$(PYTHON) -m twine upload --repository testpypi dist/*

publish: check
	$(PYTHON) -m twine upload dist/*
