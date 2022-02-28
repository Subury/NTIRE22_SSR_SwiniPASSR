from contextlib import nullcontext
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import lr_scheduler

from models.model_base import ModelBase
from models.select_network import define_G

from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip

class ModelPlain2(ModelBase):

    def __init__(self, opt):
        super(ModelPlain2, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.netG = define_G(opt)

        # ------------------------------------
        # Load Pretrained Parameters
        # ------------------------------------
        if self.opt['train']['pretrained']:
            self.pretrained = torch.load(self.opt['train']['pretrained'], map_location='cpu')
            for key in list(self.pretrained.keys()):
                if key in self.opt['train']['param_keys']:
                    self.pretrained.pop(key)

            model_dict = self.netG.state_dict()
            model_dict.update(self.pretrained)
            self.netG.load_state_dict(model_dict)
            print("Load pretrained model ", self.opt['train']['pretrained'])

        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()
    
    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key=self.opt_train['G_param_keys'])
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'], param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()
    
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)

    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)

    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss().to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']

    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        if self.opt_train['G_optimizer_type'] == 'adam':
            self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'],
                                    betas=self.opt_train['G_optimizer_betas'],
                                    weight_decay=self.opt_train['G_optimizer_wd'])
        else:
            raise NotImplementedError

    def define_scheduler(self):
        if self.opt_train['G_scheduler_type'] == 'MultiStepLR':
            self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                            self.opt_train['G_scheduler_milestones'],
                                                            self.opt_train['G_scheduler_gamma']
                                                            ))
        elif self.opt_train['G_scheduler_type'] == 'CosineAnnealingWarmRestarts':
            self.schedulers.append(lr_scheduler.CosineAnnealingWarmRestarts(self.G_optimizer,
                                                            self.opt_train['G_scheduler_periods'],
                                                            self.opt_train['G_scheduler_restart_weights'],
                                                            self.opt_train['G_scheduler_eta_min']
                                                            ))
        else:
            raise NotImplementedError

    def feed_data(self, data, need_H=True, swap=False):
        if swap:
            self.L_Left, self.L_Right = data['L_Right'].to(self.device), data['L_Left'].to(self.device)
            if need_H:
                self.H_Left, self.H_Right = data['H_Right'].to(self.device), data['H_Left'].to(self.device)
        else:
            self.L_Left, self.L_Right = data['L_Left'].to(self.device), data['L_Right'].to(self.device)
            if need_H:
                self.H_Left, self.H_Right = data['H_Left'].to(self.device), data['H_Right'].to(self.device)

    def netG_forward(self):
        self.E, (self.M_right_to_left, self.M_left_to_right), \
        (self.M_left_right_left, self.M_right_left_right), \
        (self.V_left_to_right, self.V_right_to_left) = self.netG(self.L_Left, self.L_Right)

    def optimize_parameters(self, current_step):
        
        my_context = self.netG.no_sync if self.opt['rank'] != -1 and current_step % self.opt['repeat_step'] != 0 else nullcontext

        with my_context():
            b, c, h, w = self.L_Left.shape
            self.netG_forward()
            G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H_Left)

            ### loss_smoothness
            loss_h = self.G_lossfn(self.M_right_to_left[:, :-1, :, :], self.M_right_to_left[:, 1:, :, :]) + \
                     self.G_lossfn(self.M_left_to_right[:, :-1, :, :], self.M_left_to_right[:, 1:, :, :])
            loss_w = self.G_lossfn(self.M_right_to_left[:, :, :-1, :-1], self.M_right_to_left[:, :, 1:, 1:]) + \
                     self.G_lossfn(self.M_left_to_right[:, :, :-1, :-1], self.M_left_to_right[:, :, 1:, 1:])
            loss_smooth = loss_w + loss_h

            ### loss_cycle
            Identity = torch.autograd.Variable(torch.eye(w, w).repeat(b, h, 1, 1), requires_grad=False).to(self.L_Left.device)
            loss_cycle = self.G_lossfn(self.M_left_right_left * self.V_left_to_right.permute(0, 2, 1, 3), Identity * self.V_left_to_right.permute(0, 2, 1, 3)) + \
                         self.G_lossfn(self.M_right_left_right * self.V_right_to_left.permute(0, 2, 1, 3), Identity * self.V_right_to_left.permute(0, 2, 1, 3))

            ### loss_photometric
            LR_right_warped = torch.bmm(self.M_right_to_left.contiguous().view(b*h,w,w), self.L_Left.permute(0,2,3,1).contiguous().view(b*h, w, c))
            LR_right_warped = LR_right_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            LR_left_warped = torch.bmm(self.M_left_to_right.contiguous().view(b * h, w, w), self.L_Right.permute(0, 2, 3, 1).contiguous().view(b * h, w, c))
            LR_left_warped = LR_left_warped.view(b, h, w, c).contiguous().permute(0, 3, 1, 2)

            loss_photo = self.G_lossfn(self.L_Left * self.V_left_to_right, LR_right_warped * self.V_left_to_right) + \
                          self.G_lossfn(self.L_Right * self.V_right_to_left, LR_left_warped * self.V_right_to_left)

            G_loss = G_loss + 0.005 * (loss_smooth + loss_cycle + loss_photo)
            G_loss.backward()

        if current_step % self.opt['repeat_step'] == 0:
            # ------------------------------------
            # clip_grad
            # ------------------------------------
            # `clip_grad_norm` helps prevent the exploding gradient problem.
            G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
            if G_optimizer_clipgrad > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

            self.G_optimizer.step()
            self.G_optimizer.zero_grad()

            # ------------------------------------
            # regularizer
            # ------------------------------------
            G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
            if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
                self.netG.apply(regularizer_orth)
            G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
            if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
                self.netG.apply(regularizer_clip)
            
            if self.opt_train['E_decay'] > 0:
                self.update_E(self.opt_train['E_decay'])

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['G_loss'] = G_loss.item()
    
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()
    
    def testx8(self):
        self.netG.eval()
        with torch.no_grad():
            self.E = test_mode(self.netG, self.L, mode=3, sf=self.opt['scale'], modulo=1)
        self.netG.train()
    
    def current_log(self):
        return self.log_dict
    
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L_Left.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H_Left.detach()[0].float().cpu()
        return out_dict
    
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict
    
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)
    
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)
    
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg
    
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg