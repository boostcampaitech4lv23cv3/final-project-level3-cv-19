run:
	python -m streamlit run app/main.py --server.port=30001 --server.fileWatcherType none

server:
	python app/live_server.py

step2:
	apt-get update
	apt-get install make git cmake pkg-config mingw-w64 build-essential apt-transport-https ca-certificates yasm libtool libc6 libc6-dev unzip wget libnuma1 libnuma-dev -y
	update-ca-certificates
	mkdir ~/nvidia/ && cd ~/nvidia/
	git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
	cd nv-codec-headers && make install
	cd ~/
	apt-get install ffmpeg -y