import os
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


from option import parse_args, get_model_f_params
from data import cifar10_png_wo_norm
from resnet import resnet56
from vgg import vgg16_bn
from model import Conv, UNet
from utils import *



def test(params_img, test_loader, model_h, model_ST, device, rotation):
    
    top1 = AverageMeter()
    top5 = AverageMeter()
    model_ST.eval()
    model_h.eval()
    
    batch_size = 100
    attacks = params_img['attacks']
    test_ST_output_img = params_img['test_ST_output_img']
    model_ST_select = params_img['model_ST_select']
 
    if test_ST_output_img:
        save_img = []

    with torch.no_grad():

        for k in range(len(attacks)):
            top1.reset()
            top5.reset()
            
            for i, (imgs, label) in enumerate(test_loader):

                img_w_adv = torch.chunk(imgs, len(attacks), dim=3)
                img = img_w_adv[k]
                img, label = img.to(device), label.to(device)

                adv_ST = model_ST(img)
                if test_ST_output_img:
                    save_img.append(torch.cat((img[0].unsqueeze(0), adv_ST[0].unsqueeze(0)), dim=3))

                output = model_h(adv_ST)
                prec1, prec5 = accuracy(output, label, topk=(1, 5))

                top1.update(prec1[0], img.size(0))
                top5.update(prec5[0], img.size(0))
            
            if test_ST_output_img:
                for idx in range(len(save_img)):
                    plt.imsave('{}.png'.format(idx), save_img[idx].squeeze().permute(1,2,0).cpu().numpy())


            top1_acc = top1.avg.data.cpu().numpy()[()]
            top5_acc = top5.avg.data.cpu().numpy()[()]
            logging.info("-----------------------------------------------------------------------")
            logging.info("Test on %s data: [top1: %.4f] | [top5: %.4f]" % (attacks[k], top1_acc, top5_acc))


def main():

    # -----------------------------parameters-------------------------------- #
    opt = parse_args()
    set_random_seeds(opt.random_seed)
    device = torch.device('cuda:{}'.format(opt.gpu))
    model_params = get_model_f_params(opt) # for ResNet56
    # refine the save_model_path
    curr_exp = 'evaluation'
 
    opt.save_model_path = opt.save_model_path +  curr_exp + '/'
    if not os.path.exists(opt.save_model_path):
        os.makedirs(opt.save_model_path)
    
    # save the option
    # with open('%s/opt.log' % opt.save_model_path, "w") as f:
    #     for k, v in opt.__dict__.items():
    #         f.write(str(k) + ": " + str(v) + "\n")

    # config logging file
    logger_file = os.path.join(opt.save_model_path, 'log_test.txt')
    handlers = [logging.FileHandler(logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)
    
    # -------------------------------data------------------------------------ #

    train_loader, test_loader = cifar10_png_wo_norm(opt.dataroot, opt.batch_size)
    # train_loader, test_loader = cifar100_png_wo_norm(opt.dataroot, opt.batch_size)
    # ------------------------------model------------------------------------ #
    model_h = resnet56()
    # model_h = vgg16_bn()
    if isinstance(model_h, nn.DataParallel):
        model_h = nn.DataParallel(model_h).to(device)
    else:
        model_h = model_h.to(device)
    model_h.load_state_dict(torch.load(opt.model_h_path, map_location="cuda:{}".format(opt.gpu))) # load the pretrained model_h
    model_h.eval()


    
    if opt.model_ST_select == 'UNet_w_STN':
        block = Conv
        fwd_out = [64, 128, 256, 256, 256]
        num_fwd = [2, 3, 3, 3, 3]
        back_out = [64, 128, 256, 256]
        num_back = [2, 3, 3, 3]
        use_Single_ST = True
        model_ST = UNet(opt.img_size, opt.img_size, block, 3,fwd_out, num_fwd, back_out, num_back, use_Single_ST).to(device)
        model_ST.load_state_dict(torch.load(opt.model_ST_path, map_location="cuda:{}".format(opt.gpu)), strict=False)
    else:
        assert False


    # ------------------------------train------------------------------------ #

    params_img = {}
    params_img['batch_size'] = opt.batch_size
    params_img['channel'] = opt.channel
    params_img['img_size'] = opt.img_size
    params_img['attacks'] = ['clean', 'FGSM', 'IFGSM', 'PGD10', 'PGD100', 'DeepFool', 'CW']
    params_img['test_ST_output_img'] = opt.test_ST_output_img
    
    params_img['model_ST_select'] = opt.model_ST_select

    test(params_img, test_loader, model_h, model_ST, device, opt.rotation)
    


if __name__ == '__main__':
    main()