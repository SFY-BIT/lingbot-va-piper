# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_libero_probe_train_cfg import va_libero_probe_train_cfg

va_libero_probe_lowmem_train_cfg = EasyDict(__name__='Config: VA LIBERO probe low-memory train')
va_libero_probe_lowmem_train_cfg.update(va_libero_probe_train_cfg)

# Keep the probe run small enough to survive FSDP all-gather spikes during backward.
va_libero_probe_lowmem_train_cfg.train_window_size = 4
va_libero_probe_lowmem_train_cfg.num_steps = 20
va_libero_probe_lowmem_train_cfg.gc_interval = 1
