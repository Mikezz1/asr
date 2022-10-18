sudo apt-get update
sudo apt-get install build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev
sudo apt-get install libboost-all-dev libeigen3-dev


git clone https://github.com/kpu/kenlm

cd kenlm
mkdir build
cd build

cmake ..
make -j 4
sudo make install

cd ..
python setup.py install

wget -c https://openslr.elda.org/resources/11/3-gram.pruned.3e-7.arpa.gz
gzip -d 3-gram.pruned.3e-7.arpa.gz


wget -c https://openslr.elda.org/resources/11/4-gram.arpa.gz
gzip -d 4-gram.arpa.gz

