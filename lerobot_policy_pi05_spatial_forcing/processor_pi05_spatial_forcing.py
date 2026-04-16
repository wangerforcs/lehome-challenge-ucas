from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors


def make_pi05_spatial_forcing_pre_post_processors(config, dataset_stats=None):
    return make_pi05_pre_post_processors(config=config, dataset_stats=dataset_stats)
