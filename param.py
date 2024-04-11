import argparse

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_data_path', nargs='+', type=int, default=['./Data1', './Data2'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--save_each', type=int, default=0, 
        help='save model each n epoch, 0 for not save')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument(
        '--lr_scheduler', 
        type=str, 
        default='step',
        choices=['step', 'multiStep', 'cosine', 'exp', 'cyclic']
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='unet++',
        choices=['unet', 'unet++', 'manet', 'fpn', 'deeplabv3', 'deeplabv3+',
            'pspnet', 'pan', 'panpp']
    )
    parser.add_argument(
        '--optimizer', 
        type=str, 
        default='adam',
        choices=['sgd', 'adam', 'adamw']
    )
    parser.add_argument(
        '--loss_func', 
        type=str, 
        default='dice',
        choices=['dice', 'bce', 'focal', 'dice+bce', 'dice+focal']
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='LC',
        choices=['LC', 'CN', 'FS']
    )

    parser.add_argument('--encoder', type=str, default='resnext50_32x4d')
    parser.add_argument('--encoder_weights', type=str, default='imagenet')
    parser.add_argument('--activation', type=str, default='sigmoid')

    args = parser.parse_args()

    return args