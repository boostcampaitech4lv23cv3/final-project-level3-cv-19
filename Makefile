run:
	python -m streamlit run app/main.py --server.port=30001 --server.fileWatcherType none
server:
	python app/live_server.py