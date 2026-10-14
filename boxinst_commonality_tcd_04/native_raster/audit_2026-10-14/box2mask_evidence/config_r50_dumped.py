dataset_type = 'CocoDataset'
data_root = '/data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='GenerateBoxMask'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Resize',
        img_scale=(1024, 1024),
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=(1024, 1024),
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(
        type='FilterAnnotations',
        min_gt_bbox_wh=(1e-05, 1e-05),
        keep_empty=True),
    dict(
        type='Pad',
        size=(1024, 1024),
        pad_val=dict(img=(128, 128, 128), masks=0, seg=255)),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle', img_to_float=True),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                size_divisor=32,
                pad_val=dict(img=(128, 128, 128), masks=0, seg=255)),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type='CocoDataset',
        classes=('tree', ),
        ann_file='/dt2/data/train/coco_boxonly_black.json',
        img_prefix='/dt2/data/train/images_canopyblack/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
            dict(type='GenerateBoxMask'),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Resize',
                img_scale=(1024, 1024),
                ratio_range=(0.1, 2.0),
                multiscale_mode='range',
                keep_ratio=True),
            dict(
                type='RandomCrop',
                crop_size=(1024, 1024),
                crop_type='absolute',
                recompute_bbox=True,
                allow_negative_crop=True),
            dict(
                type='FilterAnnotations',
                min_gt_bbox_wh=(1e-05, 1e-05),
                keep_empty=True),
            dict(
                type='Pad',
                size=(1024, 1024),
                pad_val=dict(img=(128, 128, 128), masks=0, seg=255)),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle', img_to_float=True),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ],
        filter_empty_gt=False),
    val=dict(
        type='CocoDataset',
        classes=('tree', ),
        ann_file='/dt2/data/val/coco_boxonly.json',
        img_prefix='/dt2/data/val/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Pad',
                        size_divisor=32,
                        pad_val=dict(img=(128, 128, 128), masks=0, seg=255)),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        filter_empty_gt=True),
    test=dict(
        type='CocoDataset',
        classes=('tree', ),
        ann_file='/dt2/data/val/coco_boxonly.json',
        img_prefix='/dt2/data/val/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Pad',
                        size_divisor=32,
                        pad_val=dict(img=(128, 128, 128), masks=0, seg=255)),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        filter_empty_gt=True))
evaluation = dict(
    interval=1673, metric=['bbox'], save_best='bbox_mAP_50', rule='greater')
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(
        type='APEarlyStop',
        interval=1673,
        patience=4,
        min_iters=16730,
        priority='VERY_LOW'),
    dict(
        type='WallClockGuard',
        max_hours=7.0,
        boundary=1673,
        priority='VERY_LOW')
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/vol/boxinst/ckpt/box2mask_r50_lsj_8x2_50e_coco_nocls.pth'
resume_from = None
workflow = [('train', 1673)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
num_thing_classes = 1
num_stuff_classes = 0
num_classes = 1
model = dict(
    type='Box2Mask',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.pytorch.org/models/resnet50-11ad3fa6.pth')),
    panoptic_head=dict(
        type='Box2MaskHead',
        in_channels=[256, 512, 1024, 2048],
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=1,
        num_stuff_classes=0,
        num_queries=100,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=2048,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0, 0.1]),
        loss_mask=dict(type='LevelsetLoss', loss_weight=1.0),
        loss_box=dict(type='BoxProjectionLoss', loss_weight=5.0)),
    panoptic_fusion_head=dict(
        type='MaskFormerFusionHead',
        num_things_classes=1,
        num_stuff_classes=0,
        loss_panoptic=None,
        init_cfg=None),
    train_cfg=dict(
        assigner=dict(
            type='MaskHungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=2.0),
            dice_cost=dict(
                type='BoxMatchingCost', weight=5.0, pred_act=True, eps=1.0)),
        sampler=dict(type='MaskPseudoSampler')),
    test_cfg=dict(
        panoptic_on=False,
        semantic_on=False,
        instance_on=True,
        max_per_image=100,
        iou_thr=0.8,
        filter_low_score=True),
    init_cfg=None)
image_size = (1024, 1024)
pad_cfg = dict(img=(128, 128, 128), masks=0, seg=255)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    eps=1e-08,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1, decay_mult=1.0),
            query_embed=dict(lr_mult=1.0, decay_mult=0.0),
            query_feat=dict(lr_mult=1.0, decay_mult=0.0),
            level_embed=dict(lr_mult=1.0, decay_mult=0.0)),
        norm_decay_mult=0.0))
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=8,
    grad_clip=dict(max_norm=0.01, norm_type=2))
lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=False,
    step=[26768, 28999],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=1.0,
    warmup_iters=10)
max_iters = 30114
runner = dict(type='IterBasedRunner', max_iters=30114)
interval = 5000
checkpoint_config = dict(
    by_epoch=False, interval=1673, save_last=True, max_keep_ckpts=2)
find_unused_parameters = True
work_dir = '/vol/boxinst/box2mask_s0'
_eval_every = 1673
_sched = dict(
    runner='iter',
    epochs=9,
    iters_per_epoch=3346,
    eval_every=1673,
    fp16=False,
    total_iters=30114,
    micro_batch=2,
    cumulative_iters=8,
    effective_batch=16,
    optimizer='AdamW',
    lr=0.0001,
    decay_at=[26768, 28999],
    num_queries=100,
    max_per_image=100,
    load_from='/vol/boxinst/ckpt/box2mask_r50_lsj_8x2_50e_coco_nocls.pth',
    method='box2mask')
seed = 0
gpu_ids = [0]
device = 'cuda'
