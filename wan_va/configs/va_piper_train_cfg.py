# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .va_piper_cfg import va_piper_cfg

va_piper_train_cfg = EasyDict(__name__='Config: VA PIPER train')
va_piper_train_cfg.update(va_piper_cfg)

va_piper_train_cfg.dataset_path = os.environ.get(
    'PIPER_DATASET_PATH',
    '/data/group1/junjie008/piper_lingbot_va',
)
va_piper_train_cfg.empty_emb_path = os.path.join(
    va_piper_train_cfg.dataset_path,
    'empty_emb.pt',
)

va_piper_train_cfg.enable_wandb = True
va_piper_train_cfg.num_init_worker = 1
va_piper_train_cfg.load_worker = 8
va_piper_train_cfg.save_interval = 200
va_piper_train_cfg.gc_interval = 50
va_piper_train_cfg.cfg_prob = 0.1

# Piper joint4 (index=3) is constant in this dataset.
va_piper_train_cfg.action_loss_mask_indices = (3,)

# Low-memory overrides (training critical)
va_piper_train_cfg.attn_window = 16
va_piper_train_cfg.frame_chunk_size = 2
va_piper_train_cfg.train_chunk_size = 1
va_piper_train_cfg.train_window_size = 1
va_piper_train_cfg.height = 128
va_piper_train_cfg.width = 128

# Sampling params (not important for train memory, keep lightweight)
va_piper_train_cfg.num_inference_steps = 5
va_piper_train_cfg.video_exec_step = -1
va_piper_train_cfg.action_num_inference_steps = 10

# Training parameters
va_piper_train_cfg.learning_rate = 1e-5
va_piper_train_cfg.beta1 = 0.9
va_piper_train_cfg.beta2 = 0.95
va_piper_train_cfg.weight_decay = 0.1
va_piper_train_cfg.warmup_steps = 100
va_piper_train_cfg.batch_size = 1
va_piper_train_cfg.gradient_accumulation_steps = 1
va_piper_train_cfg.num_steps = 5000
