from .blender import BlenderDataset
from .llff import LLFFDataset
from .llff_cls import LLFFClsDataset, LLFFClsDatasetImgBatch

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'llff_cls': LLFFClsDataset,
                'llff_cls_ib': LLFFClsDatasetImgBatch,

                }