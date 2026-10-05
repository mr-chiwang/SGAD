# Default region-preprocessing configuration.
preprocess_configs = {
    "min_area_size": 9000,
    "max_wh_ratio": 5.0,
    'topK': 3,
    'min_dist_thd': 100,
}

areagraph_configs = {
    'preprocesser_config': preprocess_configs,
    'W': 1296,
    'H': 968,
    'fs_overlap_thd': 0.8,
    'level_num': 4,
    'level_step': [560, 480, 256, 130, 0],
    'show_flag': 0,
}
