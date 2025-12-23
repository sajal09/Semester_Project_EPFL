module load gcc/11.3.0
module load python/3.10.4

virtualenv --system-site-packages venvs/venv-trial_2_3

source venv-trial_2_3/bin/activate

pip install --upgrade pip
pip install packaging --upgrade
pip install torch==2.8.0
pip install --no-cache-dir  torch_geometric
pip install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://pytorch-geometric.com/whl/torch-2.8.0%2Bcu128.html
pip install --no-cache-dir sgmllib3k
pip install --no-cache-dir pytorch-lightning
pip install --no-cache-dir h5py
pip install --no-cache-dir scikit-learn
pip install --no-cache-dir wandb

git clone https://github.com/Amber-MD/pytraj

cd pytraj

module load openblas/0.3.20

export LDFLAGS="-L/ssoft/spack/syrah/v2/opt/spack/linux-rhel9-skylake_avx512/gcc-11.3.0/openblas-0.3.20-koktto3w5c7snb56g4x3nzjeelwtmrnz/lib"
export CPPFLAGS="-I/ssoft/spack/syrah/v2/opt/spack/linux-rhel9-skylake_avx512/gcc-11.3.0/openblas-0.3.20-koktto3w5c7snb56g4x3nzjeelwtmrnz/include"
export LD_LIBRARY_PATH="/ssoft/spack/syrah/v2/opt/spack/linux-rhel9-skylake_avx512/gcc-11.3.0/openblas-0.3.20-koktto3w5c7snb56g4x3nzjeelwtmrnz/lib:$LD_LIBRARY_PATH"

python ./setup.py install

