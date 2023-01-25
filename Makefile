run_server:
	python -m gpu_dummy

run_client:
	python -m streamlit run app/main.py

run_app:	run_client	run_server