import ray


def initialize_ray(**kwargs) -> None:
    # This has to be called before PyTorch is imported.
    # https://discuss.ray.io/t/low-cpu-utilization-when-compared-to-multiprocessing/10650
    ray.init(object_store_memory=2000000000, **kwargs)
