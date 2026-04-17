# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_libero_cfg import va_libero_cfg
import os

va_libero_train_cfg = EasyDict(__name__='Config: VA LIBERO train')
va_libero_train_cfg.update(va_libero_cfg)

# 先按论文里的单个 LIBERO suite 训练，这里默认指向我们已经重建完成的
# LIBERO-Spatial LeRobot + latent 数据集。
va_libero_train_cfg.dataset_path = os.environ.get(
    'LIBERO_DATASET_PATH',
    '/mnt/hdd/sfy/datasets/libero_spatial_lerobot',
)
va_libero_train_cfg.empty_emb_path = os.path.join(va_libero_train_cfg.dataset_path, 'empty_emb.pt')
va_libero_train_cfg.enable_wandb = True
va_libero_train_cfg.load_worker = 1
va_libero_train_cfg.save_interval = 1000
va_libero_train_cfg.gc_interval = 50
va_libero_train_cfg.cfg_prob = 0.1

# Piper joint3 is constant in this dataset. Exclude aligned dim index 3 from
# action loss to avoid learning signal pollution on a non-informative channel.
va_libero_train_cfg.action_loss_mask_indices = (3,)

# Training parameters
va_libero_train_cfg.learning_rate = 1e-5
va_libero_train_cfg.beta1 = 0.9
va_libero_train_cfg.beta2 = 0.95
va_libero_train_cfg.weight_decay = 0.1
va_libero_train_cfg.warmup_steps = 10
va_libero_train_cfg.batch_size = 1
va_libero_train_cfg.gradient_accumulation_steps = 1
# 论文中 LIBERO 微调使用 4K steps，这里先对齐为 4000。
va_libero_train_cfg.num_steps = 4000
