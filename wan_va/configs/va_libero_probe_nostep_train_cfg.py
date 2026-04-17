# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_libero_probe_train_cfg import va_libero_probe_train_cfg

va_libero_probe_nostep_train_cfg = EasyDict(__name__='Config: VA LIBERO probe no-step train')
va_libero_probe_nostep_train_cfg.update(va_libero_probe_train_cfg)

# 只做前向和反向，用来测单卡资源上限，不执行优化器更新。
va_libero_probe_nostep_train_cfg.enable_wandb = False
va_libero_probe_nostep_train_cfg.no_optimizer_step = True
va_libero_probe_nostep_train_cfg.num_steps = 20
va_libero_probe_nostep_train_cfg.save_interval = 100
