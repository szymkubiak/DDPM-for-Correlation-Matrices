import datetime
configs = {
    'futures': {
        'dataset_name': 'futures',
        'train_end_dt': datetime.date(2015,12,31),
        'batch_size': 64,
        'init_dim': 32,
        'dim_mults': (1, 2, 4),
        'init_conv': (7, 5),
        'cut_output_dim_h': 3,
        'cut_output_dim_v': 3,
        'conditional': False,
    },

    'futures_cond': {
        'dataset_name': 'futures',
        'train_end_dt': datetime.date(2015, 12, 31),
        'batch_size': 64,
        'init_dim': 32,
        'dim_mults': (1, 2, 4),
        'init_conv': (7, 5),
        'cut_output_dim_h': 3,
        'cut_output_dim_v': 3,
        'conditional': True,
    },

    'fixed income': {
            'dataset_name': 'fixed income',
            'train_end_dt': datetime.date(2017,3,31),
            'batch_size': 64,
            'init_dim': 32,
            'dim_mults': (1, 2, 4, 8),
            'init_conv': (7, 5),
            'cut_output_dim_h': 4,
            'cut_output_dim_v': 4,
            'conditional': False,
    },

    'fixed income_cond': {
        'dataset_name': 'fixed income',
        'train_end_dt': datetime.date(2017, 3, 31),
        'batch_size': 64,
        'init_dim': 32,
        'dim_mults': (1, 2, 4, 8),
        'init_conv': (7, 5),
        'cut_output_dim_h': 4,
        'cut_output_dim_v': 4,
        'conditional': True,
    },

    'stocks': {
        'dataset_name': 'stocks',
        'train_end_dt': datetime.date(2013,12,31),
        'batch_size': 64,
        'init_dim': 16,
        'dim_mults': (1, 2, 4, 8, 16),
        'init_conv': (8, 8),
        'cut_output_dim_h': 9,
        'cut_output_dim_v': 9,
        'conditional': False,
    },

    'stocks_cond': {
        'dataset_name': 'stocks',
        'train_end_dt': datetime.date(2013, 12, 31),
        'batch_size': 64,
        'init_dim': 16,
        'dim_mults': (1, 2, 4, 8, 16),
        'init_conv': (8, 8),
        'cut_output_dim_h': 9,
        'cut_output_dim_v': 9,
        'conditional': True,
    }
}