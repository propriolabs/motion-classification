#Setting up on Ubuntu 14.04
sudo apt-get update

sudo apt-get install -y \
autotools-dev      \
blt-dev            \
bzip2              \
dpkg-dev           \
g++-multilib       \
gcc-multilib       \
libbluetooth-dev   \
libbz2-dev         \
libexpat1-dev      \
libffi-dev         \
libffi6            \
libffi6-dbg        \
libgdbm-dev        \
libgpm2            \
libncursesw5-dev   \
libreadline-dev    \
libsqlite3-dev     \
libssl-dev         \
libtinfo-dev       \
mime-support       \
net-tools          \
netbase            \
python-crypto      \
python-mox3        \
python-pil         \
python-ply         \
quilt              \
tk-dev             \
zlib1g-dev

sudo apt-get install python-virtualenv libblas-dev liblapack-dev libatlas-base-dev gfortran

wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda-2.2.0-Linux-x86_64.sh
bash Anaconda-2.2.0-Linux-x86_64.sh -b

# Force Python 2.7
sudo rm /usr/bin/python /usr/bin/pip
sudo ln -s $HOME/anaconda/bin/python /usr/bin/python
sudo ln -s $HOME/anaconda/bin/ipython /usr/bin/ipython
sudo ln -s $HOME/anaconda/bin/pip /usr/bin/pip
sudo sed -i '1c\#!/usr/bin/python2.6' /usr/bin/yum
export PATH=$HOME/anaconda/bin:$PATH
git clone https://github.com/mattgroh/proprio-motion-classification
#Enter credentials: user secret...
cd proprio-motion-classification
pip install -r requirements
