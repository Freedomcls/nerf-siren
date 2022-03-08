from .blender import BlenderDataset
from .llff import LLFFDataset
from .llff_cls import LLFFClsDataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'llff_cls': LLFFClsDataset}