import os
import tqdm

import torch
import numpy as np
import torch.nn as nn

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint

from utils import misc


class Trainer(object):
    def __init__(self,
                 cfg,
                 cfg_model,
                 model,
                 ema_model,
                 optimizer,
                 train_loader,
                 test_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger,
                 loss,
                 model_name):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.best_result = 0
        self.best_epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detr_loss = loss
        self.model_name = model_name
        self.output_dir = os.path.join(cfg['save_path'], model_name)
        self.tester = None
        self.cfg_model = cfg_model
        self.ema_model = ema_model

        # loading pretrain/resume model
        if cfg.get('pretrain_model'):
            assert os.path.exists(cfg['pretrain_model'])
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=cfg['pretrain_model'],
                            map_location=self.device,
                            logger=self.logger)

        if cfg.get('resume_model', None):
            resume_model_path = os.path.join(self.output_dir, "checkpoint.pth")
            assert os.path.exists(resume_model_path)
            self.epoch, self.best_result, self.best_epoch = load_checkpoint(
                model=self.model.to(self.device),
                optimizer=self.optimizer,
                filename=resume_model_path,
                map_location=self.device,
                logger=self.logger)
            self.lr_scheduler.last_epoch = self.epoch - 1
            self.logger.info("Loading Checkpoint... Best Result:{}, Best Epoch:{}".format(self.best_result, self.best_epoch))
        
    def train(self):
        start_epoch = self.epoch
        init_loss = self.compute_init_loss()
        reduce_loss = self.select_loss(init_loss)
        dynamic_loss_weightor = Hierarchical_Task_Learning(reduce_loss, self.cfg_model)

        progress_bar = tqdm.tqdm(range(start_epoch, self.cfg['max_epoch']), dynamic_ncols=True, leave=True, desc='epochs')
        best_result = self.best_result
        best_epoch = self.best_epoch
        for epoch in range(start_epoch, self.cfg['max_epoch']):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            dyloss_weights = dynamic_loss_weightor.compute_weight(reduce_loss, self.epoch)

            log_str = 'Weights: '
            for key in sorted(dyloss_weights.keys()):
                log_str += ' %s:%.4f,' %(key[:-4], dyloss_weights[key])   
            self.logger.info(log_str)

            # train one epoch
            full_loss = self.train_one_epoch(epoch, dyloss_weights=dyloss_weights)
            reduce_loss = self.select_loss(full_loss)
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()

            # save trained model
            if (self.epoch % self.cfg['save_frequency']) == 0:
                os.makedirs(self.output_dir, exist_ok=True)
                if self.cfg['save_all']:
                    ckpt_name = os.path.join(self.output_dir, 'checkpoint_epoch_%d' % self.epoch)
                else:
                    ckpt_name = os.path.join(self.output_dir, 'checkpoint')
               
                save_checkpoint(
                    get_checkpoint_state(self.model, self.ema_model, self.optimizer, self.epoch, best_result, best_epoch),
                    ckpt_name)

                if self.tester is not None:
                    self.logger.info("Test Epoch {}".format(self.epoch))
                    self.tester.inference()
                    cur_result = self.tester.evaluate()
                    if cur_result > best_result:
                        best_result = cur_result
                        best_epoch = self.epoch
                        ckpt_name = os.path.join(self.output_dir, 'checkpoint_best')
                        save_checkpoint(
                            get_checkpoint_state(self.model, self.ema_model, self.optimizer, self.epoch, best_result, best_epoch),
                            ckpt_name)
                    self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))

            progress_bar.update()

        self.logger.info("Best Result:{}, epoch:{}".format(best_result, best_epoch))

        return None

    def train_one_epoch(self, epoch, dyloss_weights=None):
        torch.set_grad_enabled(True)
        self.model.train()
        stat_dict = {}
        print(">>>>>>> Epoch:", str(epoch) + ":")

        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=(self.epoch+1 == self.cfg['max_epoch']), desc='iters')
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            for key in targets.keys():
                targets[key] = targets[key].to(self.device)
            img_sizes = targets['img_size']
            targets = self.prepare_targets(targets, inputs.shape[0])
            ##dn
            dn_args = None
            if self.cfg["use_dn"]:
                dn_args=(targets, self.cfg['scalar'], self.cfg['label_noise_scale'], self.cfg['box_noise_scale'], self.cfg['num_patterns'])
            ###
            # train one batch
            self.optimizer.zero_grad()
            outputs = self.model(inputs, calibs, targets, img_sizes, dn_args=dn_args)
            mask_dict=None
            #ipdb.set_trace()
            detr_losses_dict = self.detr_loss(outputs, targets, mask_dict)

            weight_dict = self.detr_loss.weight_dict
            detr_losses_dict_weighted = [detr_losses_dict[k] * weight_dict[k] * dyloss_weights[k] 
                                            for k in detr_losses_dict.keys() if k in weight_dict]
            detr_losses = sum(detr_losses_dict_weighted)

            detr_losses_dict = misc.reduce_dict(detr_losses_dict)
            detr_losses_dict_log = {}
            detr_losses_log = 0
            for k in detr_losses_dict.keys():
                if k in weight_dict:
                    detr_losses_dict_log[k] = (detr_losses_dict[k] * weight_dict[k] * dyloss_weights[k]).item()
                    detr_losses_log += detr_losses_dict_log[k]
            detr_losses_dict_log["loss_detr"] = detr_losses_log

            flags = [True] * 5
            if batch_idx % 30 == 0:
                print("----", batch_idx, "----")
                print("%s: %.2f, " %("loss_detr", detr_losses_dict_log["loss_detr"]))
                for key, val in detr_losses_dict_log.items():
                    if key == "loss_detr":
                        continue
                    if "0" in key or "1" in key or "2" in key or "3" in key or "4" in key or "5" in key:
                        if flags[int(key[-1])]:
                            print("")
                            flags[int(key[-1])] = False
                    print("%s: %.2f, " %(key, val), end="")
                print("")
                print("")

            detr_losses.backward()
            self.optimizer.step()

            if self.ema_model != None:
                self.ema_model.update(self.model)
            
            trained_batch = batch_idx + 1

            # accumulate statistics
            for key in detr_losses_dict.keys():
                if key not in stat_dict.keys():
                    stat_dict[key] = 0

                if isinstance(detr_losses_dict[key], int):
                    stat_dict[key] += (detr_losses_dict[key])
                else:
                    stat_dict[key] += (detr_losses_dict[key]).detach()
            progress_bar.update()
        for key in stat_dict.keys():
            stat_dict[key] /= trained_batch
        progress_bar.close()
        return stat_dict

    def prepare_targets(self, targets, batch_size):
        targets_list = []
        mask = targets['mask_2d']

        key_list = ['labels', 'boxes', 'calibs', 'depth', 'size_3d', 'heading_bin', 'heading_res', 'boxes_3d', 'ry']
        for bz in range(batch_size):
            target_dict = {}
            for key, val in targets.items():
                if key in key_list:
                    target_dict[key] = val[bz][mask[bz]]
                if key == 'depth_map':
                    target_dict[key] = val[bz]
                if key == 'obj_region':
                    target_dict[key] = val[bz]
                if key == 'calibs_perimg':
                    target_dict[key] = val[bz]
            targets_list.append(target_dict)
        return targets_list

    def compute_init_loss(self):
        self.model.train()
        disp_dict = {}
        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=True, desc='pre-training loss stat')
        with torch.no_grad():
            for batch_idx, (inputs, calibs, targets, info) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                calibs = calibs.to(self.device)
                for key in targets.keys():
                    targets[key] = targets[key].to(self.device)
                img_sizes = targets['img_size']
                targets = self.prepare_targets(targets, inputs.shape[0])
                # dn
                dn_args = None
                if self.cfg["use_dn"]:
                    dn_args=(targets, self.cfg['scalar'], self.cfg['label_noise_scale'], self.cfg['box_noise_scale'], self.cfg['num_patterns'])

                # train one epoch
                outputs = self.model(inputs, calibs, targets, img_sizes, dn_args=dn_args)
                mask_dict=None
                detr_losses_dict = self.detr_loss(outputs, targets, mask_dict)
                
                trained_batch = batch_idx + 1
                # accumulate statistics
                for key in detr_losses_dict.keys():
                    if key not in disp_dict.keys():
                        disp_dict[key] = 0
                    disp_dict[key] += detr_losses_dict[key]      
                progress_bar.update()
            for key in disp_dict.keys():
                disp_dict[key] /= trained_batch    
            progress_bar.close()         
        return disp_dict
    
    def select_loss(self, loss_dict):
        select_loss_name = ['loss_region', 'loss_ce', 'loss_center', 'loss_giou', 'loss_bbox',
                            'loss_depth_map', 'loss_dim', 'loss_angle', 'loss_depth', 'loss_threediou', 'loss_proj']
        reduce_loss_dict = {}
        for k, v in loss_dict.items():
            if k in select_loss_name:
                reduce_loss_dict[k] = v
        return reduce_loss_dict
    

class Hierarchical_Task_Learning:
    def __init__(self, init_loss, cfg_model, stat_epoch_nums=5):
        self.cfg_model = cfg_model
        self.index2term = [*init_loss.keys()]
        self.term2index = {term:self.index2term.index(term) for term in self.index2term}  # term2index
        self.stat_epoch_nums = stat_epoch_nums
        self.past_losses=[]
        self.loss_graph = {'loss_region':[],
                           'loss_ce':[], 
                           'loss_center':[],
                           'loss_giou':[],
                           'loss_bbox':[],
                           'loss_depth_map':['loss_region'],
                           'loss_dim':['loss_center','loss_giou','loss_bbox'], 
                           'loss_angle':['loss_center','loss_giou','loss_bbox'], 
                           'loss_depth':['loss_giou','loss_bbox','loss_dim'],
                           'loss_threediou':['loss_center','loss_dim','loss_angle','loss_depth'],
                           'loss_proj':['loss_center','loss_dim','loss_depth'],
                           }
    
    @torch.no_grad()
    def compute_weight(self, current_loss, epoch):
        # total epoch
        T=250
        # compute initial weights
        loss_weights = {}
        eval_loss_input = torch.cat([_.unsqueeze(0) for _ in current_loss.values()]).unsqueeze(0)
        for term in self.loss_graph:
            if len(self.loss_graph[term])==0:
                loss_weights[term] = torch.tensor(1.0).to(current_loss[term].device)
            else:
                loss_weights[term] = torch.tensor(0.0).to(current_loss[term].device) 
        # update losses list
        if len(self.past_losses)==self.stat_epoch_nums:
            past_loss = torch.cat(self.past_losses)
            mean_diff = (past_loss[:-2]-past_loss[2:]).abs().mean(0)
            if not hasattr(self, 'init_diff'):
                self.init_diff = mean_diff
            c_weights = torch.clamp(1-(mean_diff/self.init_diff).relu().unsqueeze(0), 0.0, 1.0)
            
            time_value = min(((epoch)/(T)), 1.0)
            for current_topic in self.loss_graph:
                if len(self.loss_graph[current_topic])!=0:
                    control_weight = 1.0
                    for pre_topic in self.loss_graph[current_topic]:
                        control_weight *= c_weights[0][self.term2index[pre_topic]]
                    # loss_weights[current_topic] = time_value ** (1-control_weight)
                    if len(self.loss_graph[current_topic]) == 0: control_weight = control_weight
                    else: control_weight = control_weight ** (1/len(self.loss_graph[current_topic]))
                    loss_weights[current_topic] = time_value ** (1-control_weight)
                    if loss_weights[current_topic] != loss_weights[current_topic]:
                        for pre_topic in self.loss_graph[current_topic]:
                            print('NAN===============', time_value, control_weight, c_weights[0][self.term2index[pre_topic]], pre_topic, self.term2index[pre_topic])
            # pop first list
            self.past_losses.pop(0)
        self.past_losses.append(eval_loss_input)

        if self.cfg_model['aux_loss']:
            aux_weight_dict = {}
            for i in range(self.cfg_model['dec_layers'] - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in loss_weights.items()})
            loss_weights.update(aux_weight_dict)

        inter_weight_dict = {}
        inter_keys = ['loss_ce', 'loss_bbox', 'loss_center', 'loss_giou']
        for i in range(self.cfg_model['dec_layers']):
            inter_weight_dict.update({k + f'_inter_{i}': v for k, v in loss_weights.items() if k in inter_keys})
        loss_weights.update(inter_weight_dict)

        return loss_weights
    
    @torch.no_grad()
    def update_e0(self,eval_loss):
        self.epoch0_loss = torch.cat([_.unsqueeze(0) for _ in eval_loss.values()]).unsqueeze(0)
