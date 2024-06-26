import argparse

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_id', type=str, default='', help='Naming one training. Useful for control/del.sh.')
    parser.add_argument('--base_data_path', nargs='+', type=str, default=['./Data1', './Data2'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epoch', type=int, default=120)
    parser.add_argument('--save_each', type=int, default=0, 
        help='save model each n epoch, 0 for not save and only save the best.')
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
        choices=['unet', 'unet++', 'manet', 'linknet', 'fpn', 'pspnet', 'pan', 'deeplabv3', 'deeplabv3+']
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
    
    parser.add_argument('--gridsearch',type=int,default=0)
    args = parser.parse_args()

    return args