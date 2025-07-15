# TwinTrack Main Configuration

class Config:
    """
    Main configuration for TwinTrack experiments.
    """
    # Model
    backbone = 'resnet101'  # 'resnet50' or 'resnet101'
    out_dim = 256
    use_tlcfs = True
    use_dltc = True
    use_tel = True
    memory_len = 10
    association_factor = 0.3
    tel_lambda = 0.6
    dropout = 0.1

    # Training
    batch_size = 8
    learning_rate = 1e-4
    weight_decay = 1e-5
    epochs = 70
    warmup_epochs = 5
    lr_min = 1e-6
    optimizer = 'adamw'
    amp = True  # Mixed precision

    # Data
    dataset = 'DCT'  # 'DCT', 'AnimalTrack', 'BuckTales', 'HarvardCow'
    data_root = './data/DCT/'
    input_size = (1280, 736)
    frame_interval = 2
    num_workers = 4

    # Evaluation
    eval_interval = 1
    save_best = True
    log_dir = './logs/'
    checkpoint_dir = './checkpoints/'

    # Device
    device = 'cuda'  # or 'cpu'

    # Ablation
    ablation = None  # e.g., 'no_tlcfs', 'no_dltc', 'no_tel'

    # Visualization
    vis_save_dir = './figures/'

    # Random seed
    seed = 42 