# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_franka_cfg import va_franka_cfg
from .va_libero_cfg import va_libero_cfg
from .va_libero_probe_nostep_train_cfg import va_libero_probe_nostep_train_cfg
from .va_libero_probe_train_cfg import va_libero_probe_train_cfg
from .va_libero_probe_lowmem_train_cfg import va_libero_probe_lowmem_train_cfg
from .va_libero_smoke_train_cfg import va_libero_smoke_train_cfg
from .va_robotwin_cfg import va_robotwin_cfg
from .va_franka_i2va import va_franka_i2va_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg
from .va_libero_train_cfg import va_libero_train_cfg
from .va_piper_cfg import va_piper_cfg
from .va_piper_train_cfg import va_piper_train_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg
from .va_demo_train_cfg import va_demo_train_cfg
from .va_demo_cfg import va_demo_cfg
from .va_demo_i2va import va_demo_i2va_cfg


VA_CONFIGS = {
    'robotwin': va_robotwin_cfg,
    'libero': va_libero_cfg,
    'libero_probe_nostep_train': va_libero_probe_nostep_train_cfg,
    'libero_probe_train': va_libero_probe_train_cfg,
    'libero_probe_lowmem_train': va_libero_probe_lowmem_train_cfg,
    'libero_smoke_train': va_libero_smoke_train_cfg,
    'franka': va_franka_cfg,
    'robotwin_i2av': va_robotwin_i2va_cfg,
    'franka_i2av': va_franka_i2va_cfg,
    'robotwin_train': va_robotwin_train_cfg,
    'libero_train': va_libero_train_cfg,
    'piper': va_piper_cfg,
    'piper_train': va_piper_train_cfg,
    'demo': va_demo_cfg,
    'demo_train': va_demo_train_cfg,
    'demo_i2av': va_demo_i2va_cfg,
}
