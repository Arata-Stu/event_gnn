from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# CUDA 実装モジュール（GPUベース）
ev_graph_cuda = CUDAExtension(
    name='ev_graph_cuda',
    sources=['src/data/graph/ev_graph.cu'],
)

setup(
    name='ev_graph_cuda',
    version='1.0',
    description='Event based Graph construction in CUDA',
    ext_modules=[ev_graph_cuda],
    cmdclass={'build_ext': BuildExtension},
)
