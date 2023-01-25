run_server:
	python -m app

run_client:
	python -m streamlit run app/frontend/index.py

run_relay:
	python ./app/relay.py

run_app: run_server run_client