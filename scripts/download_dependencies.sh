# Intall dev packages
sudo apt-get update
sudo apt-get install -y build-essential cmake automake autotools-dev festival espeak-ng mbrola virtualenv 

# Download pytorch
wget -P scripts https://download.pytorch.org/whl/cu75/torch-0.1.12.post2-cp36-cp36m-linux_x86_64.whl
wget -P scripts https://download.pytorch.org/whl/cu75/torch-0.1.12.post2-cp27-none-linux_x86_64.whl
