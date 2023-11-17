# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import cv2

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmengine.model import (MMDistributedDataParallel, convert_sync_batchnorm,
                            is_model_wrapper, revert_sync_batchnorm)
from mmengine.runner.activation_checkpointing import turn_on_activation_checkpointing
from mmengine.model.efficient_conv_bn_eval import \
    turn_on_efficient_conv_bn_eval

from mmdet.utils import setup_cache_size_limit_of_dynamo
import torch
import torchvision
from torchvision.utils import save_image
import gc
from utils.mmdet_tools import inverse_to_base, compute_normalized_cross_correlation, save_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def mmdet_attack_coco_ncc(runner: Runner, ncc_thres, attack_threshold , start_iter, end_iter, path_name):
    if is_model_wrapper(runner.model):
        ori_model = runner.model.module
    else:
        ori_model = runner.model
    assert hasattr(ori_model, 'train_step'), (
        'If you want to train your model, please make sure your model '
        'has implemented `train_step`.')
    
    

    if runner._val_loop is not None:
        assert hasattr(ori_model, 'val_step'), (
            'If you want to validate your model, please make sure your '
            'model has implemented `val_step`.')

    if runner._train_loop is None:
        raise RuntimeError(
            '`self._train_loop` should not be None when calling train '
            'method. Please provide `train_dataloader`, `train_cfg`, '
            '`optimizer` and `param_scheduler` arguments when '
            'initializing runner.')

    runner._train_loop = runner.build_train_loop(
        runner._train_loop)  # type: ignore

    # `build_optimizer` should be called before `build_param_scheduler`
    #  because the latter depends on the former
    runner.optim_wrapper = runner.build_optim_wrapper(runner.optim_wrapper)
    # Automatically scaling lr by linear scaling rule
    runner.scale_lr(runner.optim_wrapper, runner.auto_scale_lr)

    if runner.param_schedulers is not None:
        runner.param_schedulers = runner.build_param_scheduler(  # type: ignore
            runner.param_schedulers)  # type: ignore
    
    if runner._val_loop is not None:
        runner._val_loop = runner.build_val_loop(
            runner._val_loop)  # type: ignore
    # TODO: add a contextmanager to avoid calling `before_run` many times
    runner.call_hook('before_run')

    # initialize the model weights
    runner._init_model_weights()

    # try to enable activation_checkpointing feature
    modules = runner.cfg.get('activation_checkpointing', None)
    if modules is not None:
        runner.logger.info(f'Enabling the "activation_checkpointing" feature'
                            f' for sub-modules: {modules}')
        turn_on_activation_checkpointing(ori_model, modules)

    # try to enable efficient_conv_bn_eval feature
    modules = runner.cfg.get('efficient_conv_bn_eval', None)
    if modules is not None:
        runner.logger.info(f'Enabling the "efficient_conv_bn_eval" feature'
                            f' for sub-modules: {modules}')
        turn_on_efficient_conv_bn_eval(ori_model, modules)

    # make sure checkpoint-related hooks are triggered after `before_run`
    runner.load_or_resume()

    # Initiate inner count of `optim_wrapper`.
    # self._train_loop.iter = 0; self._train_loop.max_iters = 703596
    runner.optim_wrapper.initialize_count_status(
        runner.model,
        runner._train_loop.iter,  # type: ignore
        runner._train_loop.max_iters)  # type: ignore

    # Maybe compile the model according to options in self.cfg.compile
    # This must be called **AFTER** model has been wrapped.
    runner._maybe_compile('train_step')

    runner.call_hook('before_train')

    # print(type(runner._train_loop))
    runner.model.train()
    
    for idx, data_batch in enumerate(runner._val_loop.dataloader):
        print(idx)
        if (idx < start_iter):
            continue
        if (idx > end_iter):
            break
        losses_hist = []
        # with(runner.optim_wrapper.optim_context(runner.model)):
        data = runner.model.data_preprocessor(data_batch, True)
    
        # img = data.squeeze(0).
        # plt.imshow(data['inputs'].cpu().squeeze().permute(1, 2, 0))
        # plt.show()
        
        data['inputs'].requires_grad_()
        data['inputs'].retain_grad()
        
        # print(data['inputs'].size())
        losses = runner.model._run_forward(data, mode='loss')
        
        parsed_losses, log_vars = runner.model.parse_losses(losses)  # type: ignore
        losses_hist.append(parsed_losses.detach().cpu())
        
        parsed_losses.retain_grad()
        parsed_losses.backward() 
        gradient = data['inputs'].grad[0].clone()
        
        ori_image = data_batch['inputs'][0].cuda()
        
        # Faster_RCNN
        if ("faster-rcnn" in path_name):
            step_size = 200
            total_steps = 2000
        # RetinaNet
        elif ("retinanet" in path_name):
            step_size = 300
            total_steps = 3000
        
        # Swin Transformer   
        elif ("swin" in path_name):
            step_size = 400
            total_steps = 3000
        
        for number_of_steps in range(total_steps):
            
            preds = runner.model._run_forward(data, mode='predict')
           
            gradient_mask = data['inputs'][0]
            flag_zero_pred = False
            for _, pred in enumerate(preds):
                labels, bboxes, scores = pred.get('pred_instances').labels, pred.get('pred_instances').bboxes, pred.get('pred_instances').scores
                
                larger_than_threhold = scores >= attack_threshold
                labels_pred = labels[larger_than_threhold]
                bboxes_pred  = bboxes[larger_than_threhold]
                scores_pred  = scores[larger_than_threhold]
 
                mask = torch.zeros_like(data['inputs'][0])
                   
                if (bboxes_pred.shape[0] == 0):
                    flag_zero_pred = True
                    break
                
                count = 0
                for box in bboxes_pred: 
                    x1,y1,x2,y2 = box
                    ncc_value = compute_normalized_cross_correlation((ori_image[:,int(y1):int(y2),int(x1):int(x2)]).cuda() , (data_batch['inputs'][0][:,int(y1):int(y2),int(x1):int(x2)]).cuda())
                    # print(ncc_value)
                    if ncc_value < ncc_thres:
                        count += 1
                        continue
                    mask[:,int(y1):int(y2),int(x1):int(x2)] = 1
                if count == bboxes_pred.shape[0]:
                    
                    flag_zero_pred = True
                    break
                
                for box in bboxes_pred:
                    x1,y1,x2,y2 = box
                    mask[:,int(y1):int(y2),int(x1):int(x2)] = 1
      
                gradient_mask = torch.where(mask == 1, data['inputs'][0] + gradient*step_size, data['inputs'][0]).cuda()
                
                
            path_save = os.path.split(data['data_samples'][0].img_path)[1]
            print('image: ',path_save)
            print('number of steps: ', number_of_steps)
            if number_of_steps == total_steps - 1 or flag_zero_pred == True:
                path_save = os.path.split(data['data_samples'][0].img_path)[1]
                save_image(inverse_to_base(gradient_mask),f'coco2017_{path_name}/{path_save}')
                break
            
            data_batch['inputs'][0] = torch.clamp((inverse_to_base(gradient_mask).flip(0)*255).detach(), 0, 255)
             
            data = runner.model.data_preprocessor(data_batch, True)
            # data['inputs']    = torch.autograd.Variable(data['inputs'])
            data['inputs'].requires_grad_()
            data['inputs'].retain_grad()
            losses = runner.model._run_forward(data, mode='loss')
            
            
            parsed_losses, log_vars = runner.model.parse_losses(losses)  # type: ignore
            losses_hist.append(parsed_losses.detach().cpu())
            print(losses_hist[-1])
            parsed_losses.retain_grad()
            parsed_losses.backward() 
            gradient = data['inputs'].grad[0].clone()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.memory_allocated()
            
def main(ncc_thres, attack_threshold, start_iter, end_iter):
    
    args = parse_args()
    path_name = os.path.split(args.config)[1][:-3]
    if os.path.exists(f"coco2017_{path_name}") == False:
        os.mkdir(f"coco2017_{path_name}")
    
    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    
    # enable automatic-mixed-precision training
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # resume is determined in this priority: resume from > auto_resume
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume


    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    mmdet_attack_coco_ncc(runner, ncc_thres, attack_threshold, start_iter, end_iter, path_name)


if __name__ == '__main__':
    
    ncc_thres           = 0.6
    conf_thres          = 0.5
    
    
    
    index = 0
    start_index = 1000*index
    end_index   = 1000*(index+1)
    
    
    main(ncc_thres, conf_thres,start_index, end_index)
    