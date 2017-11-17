#!/bin/bash
# If you want to use tensorflow-gpu on arpeggio, run this script!

cd ${HOME}

shell=`echo $SHELL | awk -F / '{print $NF}'`

git clone https://github.com/pyenv/pyenv.git ~/.pyenv
echo '# pyenv setting' >> ~/.${shell}rc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.${shell}rc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.${shell}rc
echo 'eval "$(pyenv init -)"' >> ~/.${shell}rc
source ~/.${shell}rc

pyenv install $1
pyenv local $1

echo '# cuda setting' >> ~/.${shell}rc
echo 'export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64"' >> ~/.${shell}rc
source ~/.${shell}rc

pip install tensorflow-gpu
