# EventGNN

## setup
```bash
python3.11 -m venv env
source env/bin/activate

pip3 install --upgrade setuptools wheel
pip3 install -r requirements.txt
pip3 install --no-build-isolation torch_spline_conv
pip3 install --no-build-isolation torch_scatter
pip3 install --no-build-isolation torch-cluster
pip3 install --no-build-isolation torch-sparse

## for detectron2
mkdir lib
cd lib
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
git checkout 32bd159d7263683e39bf4e87e5c4ac88bad2fd73
pip3 install -e lib/detectron2

## for event graph cuda
python3 setup.py build_ext --inplace
```