"""
Time: 2022-02-16
"""
import math
import random
import os.path
import logging
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


from models.select_model import define_Model
from data.select_dataset import define_Dataset

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

def main(json_path=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    print(opt["path"])
    opt['dist'] = parser.parse_args().dist

    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']

    if opt['rank'] == 0:
        option.save(opt)

    opt = option.dict_to_nonedict(opt)

    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    model = define_Model(opt)
    model.init_train()

    if opt['rank'] == 0:
        logger.info(model.info_network())
    
    for epoch in range(1000000):
        
        for train_data in train_loader:

            current_step += 1

            model.feed_data(train_data)

            model.optimize_parameters(current_step)

            if current_step % (opt['repeat_step'] * opt['train']['checkpoint_print']) == 0 and opt['rank'] == 0:
                logs = model.current_log()
                message = '[Train] [{:s}] iter: {:d}, lr:{:.3e}, '.format(opt['netG']['net_type'], current_step // opt['repeat_step'], model.current_learning_rate())
                for k in ['loss_sr', 'loss_photo', 'loss_smooth', 'loss_cycle', 'loss_cons', 'G_loss']:
                    message += '{:s}: {:.3e} '.format(k, sum(logs[k]) / len(logs[k]))
                    logs[k] = []
                logger.info(message)
            
            if current_step % (opt['repeat_step'] * opt['train']['checkpoint_save']) == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)
            
            if current_step % (opt['repeat_step'] * opt['train']['checkpoint_test']) == 0 and opt['rank'] == 0:

                idx, avg_psnr = 0, 0.0

                for test_data in test_loader:
                        
                    idx += 1
                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    EL_img = util.tensor2uint(visuals['EL'])
                    HL_img = util.tensor2uint(visuals['HL'])
                    ER_img = util.tensor2uint(visuals['ER'])
                    HR_img = util.tensor2uint(visuals['HR'])

                    current_psnr = 0.5 * (util.calculate_psnr(EL_img, HL_img, border=border) + util.calculate_psnr(ER_img, HR_img, border=border))

                    avg_psnr += current_psnr

                avg_psnr = avg_psnr / idx
                logger.info('[Validation] iter:{:d}, Average PSNR : {:<.2f}dB\n'.format(current_step // opt['repeat_step'], avg_psnr))

            if current_step % opt['repeat_step'] == 0:
                model.update_learning_rate(current_step)

        if current_step == 500000:
            break            

if __name__ == '__main__':
    main()