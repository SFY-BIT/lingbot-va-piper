# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_libero_train_cfg import va_libero_train_cfg

va_libero_smoke_train_cfg = EasyDict(__name__='Config: VA LIBERO smoke train')
va_libero_smoke_train_cfg.update(va_libero_train_cfg)

# Keep the smoke test lightweight and avoid external logging noise.
va_libero_smoke_train_cfg.enable_wandb = True
va_libero_smoke_train_cfg.num_init_worker = 1
va_libero_smoke_train_cfg.load_worker = 0
va_libero_smoke_train_cfg.save_interval = 5
va_libero_smoke_train_cfg.gc_interval = 5
va_libero_smoke_train_cfg.num_steps = 10
