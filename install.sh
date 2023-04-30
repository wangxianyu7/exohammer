conda create -n $1
conda activate $1
conda install pip
conda install python=3.8
pip install .
