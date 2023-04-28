import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Defense Transformer')
    parser.add_argument('--dataroot', type=str, default='datasets/cifar10_resnet56_w_adv')

    parser.add_argument('--model_ST_select', type=str, default='UNet_w_STN')
    parser.add_argument('--model_ST_path', type=str, default=' ')
    parser.add_argument('--model_h_path', type=str, default='pytorch_resnet_cifar10/best_model.th')
    parser.add_argument('--save_model_path', type=str, default='checkpoint/')
    parser.add_argument('--rotation', action = 'store_true')
    parser.add_argument('--test_ST_output_img', action = 'store_true')
    parser.add_argument('--rand_bias', type=str, default='None')

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--optimizer_select', type=str, default='adam')
    parser.add_argument('--gamma', type=float, default=0.1)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--channel', type=int, default=3)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lambda2', type=float, default=1)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--random_seed', type=int, default=2)

    args = parser.parse_args()
    
    return args
    

def get_model_f_params(opt):
    
    model_params = {}
    model_params['block_type'] = 'basic'
    model_params['num_blocks'] = [9,9,9]
    model_params['num_classes'] = opt.num_classes
    model_params['input_size'] = opt.img_size
    return model_params
