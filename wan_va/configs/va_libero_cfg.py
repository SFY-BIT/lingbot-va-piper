# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .shared_config import va_shared_cfg

va_libero_cfg = EasyDict(__name__='Config: VA LIBERO')
va_libero_cfg.update(va_shared_cfg)
va_shared_cfg.infer_mode = 'server'

va_libero_cfg.wan22_pretrained_model_name_or_path = os.environ.get(
    'LINGBOT_VA_MODEL_PATH',
    '/mnt/hdd/sfy/models/lingbot-va-base',
)

va_libero_cfg.attn_window = 72
va_libero_cfg.frame_chunk_size = 4
va_libero_cfg.env_type = 'libero'

va_libero_cfg.height = 256
va_libero_cfg.width = 256
va_libero_cfg.action_dim = 30
va_libero_cfg.action_per_frame = 4
va_libero_cfg.obs_cam_keys = [
    'observation.images.image',
    'observation.images.wrist_image',
]
va_libero_cfg.guidance_scale = 5
va_libero_cfg.action_guidance_scale = 1

va_libero_cfg.num_inference_steps = 25
va_libero_cfg.video_exec_step = -1
va_libero_cfg.action_num_inference_steps = 50

va_libero_cfg.snr_shift = 5.0
va_libero_cfg.action_snr_shift = 1.0

# The pretrained LingBot-VA transformer uses a 30D action token interface.
# LIBERO only provides 7D actions, so we map those 7 channels into the first
# 7 slots and leave the remaining slots padded / masked out.
va_libero_cfg.used_action_channel_ids = list(range(7))
inverse_used_action_channel_ids = [len(va_libero_cfg.used_action_channel_ids)
                                   ] * va_libero_cfg.action_dim
for i, j in enumerate(va_libero_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_libero_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

# Aligned action dimensions to be excluded from action loss.
# Empty by default; set in train config when a joint is known to be constant.
va_libero_cfg.action_loss_mask_indices = ()

# LIBERO actions are 7D in the downloaded LeRobot-format latent dataset:
# [x, y, z, axis_angle1, axis_angle2, axis_angle3, gripper]
# The open-source issue discussion suggests using the following 7D quantiles,
# while keeping the transformer-facing action interface padded to 30D.
va_libero_cfg.action_norm_method = 'quantiles'
va_libero_cfg.norm_stat = {
    "q01": [
        -0.7454732114076613,
        -0.6616071462631226,
        -0.9375,
        -0.1071428582072258,
        -0.20678570866584778,
        -0.1842857152223587,
        0.0,
    ] + [0.0] * (va_libero_cfg.action_dim - 7),
    "q99": [
        0.9375,
        0.8758928775787354,
        0.9321428537368774,
        0.1039285734295845,
        0.17678570747375488,
        0.14571428298950195,
        1.0,
    ] + [1.0] * (va_libero_cfg.action_dim - 7),
}
