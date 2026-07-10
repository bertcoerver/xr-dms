.PHONY: release clean build upload

release: clean build upload

clean:
	rm -rf dist/ build/ src/*.egg-info

build:
	python -m build

upload:
	twine upload dist/*
