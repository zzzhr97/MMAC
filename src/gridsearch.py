import train

from itertools import product
from types import SimpleNamespace

training_id = "zhr2"
base_data_path = ['./Data1', './Data2']
epoch = 120
save_each = 0
device = "cuda"
encoder_weights = 'imagenet'
datasets = ["CN","FS","LC"]

models = ['pan', 'manet', 'unet++']  
encoders = ['resnext50_32x4d', 'resnet101'] 

'''
seeds = [42]
batch_sizes = [100]
lrs = [0.005,0.001,0.0001]
wds = [0,1e-4,1e-5]
lr_schedulers = ['step', 'multiStep', 'cosine', 'exp', 'cyclic']
models = ['unet', 'unet++', 'manet', 'linknet', 'fpn', 'pspnet', 'pan', 'deeplabv3', 'deeplabv3+']
optimizers = ['sgd', 'adam', 'adamw']
loss_funcs = ['dice', 'bce', 'focal', 'dice+bce', 'dice+focal']
encoders = ['resnext50_32x4d','resnet101','resnet152']
activations = ['sigmoid','relu','tanh']
'''
seeds = [42]
batch_sizes = [4]
lrs = [1e-4]
wds = [0]
lr_schedulers = ['step', 'multiStep']
models = ['pan', 'manet', 'unet++']     
optimizers = ['adam']
loss_funcs = ['dice']
encoders = ['resnet101']
activations = ['sigmoid']

for (model, optimizer, loss_func, encoder, activation, lr, wd, lr_scheduler, batch_size, seed) in product(
    models, optimizers, loss_funcs, encoders, activations, lrs, wds, lr_schedulers, batch_sizes, seeds):
    for dataset in datasets:
        args = SimpleNamespace(
            training_id=training_id,
            base_data_path=base_data_path,
            epoch=epoch,
            save_each=save_each,
            device=device,
            encoder_weights=encoder_weights,
            dataset=dataset,
            seed=seed,
            batch_size=batch_size,
            lr=lr,
            wd=wd,
            lr_scheduler=lr_scheduler,
            model=model,
            optimizer=optimizer,
            loss_func=loss_func,
            encoder=encoder,
            activation=activation,
            gridsearch=1
        )

        
        trainer = train.Train(args)
        trainer.train()

