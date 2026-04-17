# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .shared_config import va_shared_cfg

va_piper_cfg = EasyDict(__name__='Config: VA PIPER')
va_piper_cfg.update(va_shared_cfg)
va_piper_cfg.infer_mode = 'server'

va_piper_cfg.wan22_pretrained_model_name_or_path = os.environ.get(
    'LINGBOT_VA_MODEL_PATH',
    '/data/group1/junjie008/piper_lingbot_va',
)

# Keep core setup aligned with current piper training choice.
va_piper_cfg.attn_window = 30
va_piper_cfg.frame_chunk_size = 4
va_piper_cfg.env_type = 'libero'

# Lower resolution to reduce training/backward memory peak.
va_piper_cfg.height = 128
va_piper_cfg.width = 128
va_piper_cfg.action_dim = 30
va_piper_cfg.action_per_frame = 4
va_piper_cfg.obs_cam_keys = [
    'observation.images.one',
    'observation.images.two',
]
va_piper_cfg.guidance_scale = 5
va_piper_cfg.action_guidance_scale = 1

# Sampling setup aligned to demo-style fast sampling.
va_piper_cfg.num_inference_steps = 5
va_piper_cfg.video_exec_step = -1
va_piper_cfg.action_num_inference_steps = 10

va_piper_cfg.snr_shift = 5.0
va_piper_cfg.action_snr_shift = 1.0

# The pretrained LingBot-VA transformer expects a 30D action interface.
# Piper actions are 7D, so map these channels into the first 7 aligned slots.
va_piper_cfg.used_action_channel_ids = list(range(7))
inverse_used_action_channel_ids = [len(va_piper_cfg.used_action_channel_ids)] * va_piper_cfg.action_dim
for i, j in enumerate(va_piper_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_piper_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

# Optional aligned action dimensions excluded from action loss.
va_piper_cfg.action_loss_mask_indices = ()

# Quantile stats computed from /mnt/hdd/sfy/piper_lingbot_va (32668 frames).
va_piper_cfg.action_norm_method = 'quantiles'
va_piper_cfg.norm_stat = {
    "q01": [
        -0.8685850501060486,
        0.022910088300704956,
        -1.4058862924575806,
        0.0,
        -0.021205764263868332,
        -1.0678727626800537,
        0.0008275953005068004,
    ] + [0.0] * (va_piper_cfg.action_dim - 7),
    "q99": [
        0.09372925758361816,
        2.0912911891937256,
        0.0,
        0.0,
        1.2190511226654053,
        0.06920868903398514,
        0.07411746680736542,
    ] + [1.0] * (va_piper_cfg.action_dim - 7),
}
