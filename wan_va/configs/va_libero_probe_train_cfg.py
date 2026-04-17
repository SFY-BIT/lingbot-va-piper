# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_libero_train_cfg import va_libero_train_cfg

va_libero_probe_train_cfg = EasyDict(__name__='Config: VA LIBERO probe train')
va_libero_probe_train_cfg.update(va_libero_train_cfg)


va_libero_probe_train_cfg.enable_wandb = False
va_libero_probe_train_cfg.load_worker = 0
va_libero_probe_train_cfg.num_init_worker = 1
va_libero_probe_train_cfg.batch_size = 1
va_libero_probe_train_cfg.gradient_accumulation_steps = 1
va_libero_probe_train_cfg.save_interval = 100
va_libero_probe_train_cfg.gc_interval = 5
va_libero_probe_train_cfg.num_steps = 20

va_libero_probe_train_cfg.train_chunk_size = 1
va_libero_probe_train_cfg.train_window_size = 8
