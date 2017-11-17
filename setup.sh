#!/bin/bash
# If you want to use tensorflow-gpu on arpeggio, run this script!

cd ${HOME}
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo '# pyenv setting' >> ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

pyenv install $1
pyenv local $1

echo '# cuda setting' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64"' >> ~/.bashrc
source ~/.bashrc

pip install tensorflow-gpu
