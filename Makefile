.PHONY: install ui bootstrap demo run watch test

install:
	pip install -r requirements.txt

ui:
	streamlit run streamlit_app.py

bootstrap:
	python scripts/bootstrap_db.py

demo:
	python scripts/ingest_demo_data.py

run:
	python scripts/run_pipeline.py --file datasets/raw_samples/sample_bank_statement.csv

watch:
	python scripts/watch_and_finetune.py --interval 20

test:
	pytest -q
