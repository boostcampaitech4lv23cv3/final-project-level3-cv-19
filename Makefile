run_server:
	python -m onnx_tensorrt

run_client:
	python -m streamlit run app/main.py --server.port=30001 --server.fileWatcherType none

run_app:	run_client	run_server
