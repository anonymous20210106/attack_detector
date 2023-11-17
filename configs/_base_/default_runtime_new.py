checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = "https://github.com/SwinTransformer/storage/releases/download/v1.0.3/mask_rcnn_swin_tiny_patch4_window7_1x.pth"
resume_from = None
workflow = [('train', 1)]