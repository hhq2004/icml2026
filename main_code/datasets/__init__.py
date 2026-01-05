from .videomme import VideoMME
# from .longvideobench import LongVideoBench
# from .mluv import MLUV
DATASET_REGISTRY = {
    "VideoMME": VideoMME,
    # "LongVideoBench": LongVideoBench,
    # "MLUV": MLUV
}