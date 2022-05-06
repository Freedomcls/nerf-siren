from .blender import BlenderDataset, BlenderDatasetWithClsBatch
from .llff import LLFFDataset
from .llff_cls import LLFFClsDataset, LLFFClsDatasetImgBatch

dataset_dict = {'blender': BlenderDataset,
                'blender_cls_ib': BlenderDatasetWithClsBatch,
                'llff': LLFFDataset,
                'llff_cls': LLFFClsDataset,
                'llff_cls_ib': LLFFClsDatasetImgBatch,

                }