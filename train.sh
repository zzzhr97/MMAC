
training_id="zhr1"

python train.py --dataset LC --model unet++ --training_id ${training_id}
python train.py --dataset CN --model unet++ --training_id ${training_id}
python train.py --dataset FS --model unet++ --training_id ${training_id}

python train.py --dataset LC --model unet --training_id ${training_id}
python train.py --dataset CN --model unet --training_id ${training_id}
python train.py --dataset FS --model unet --training_id ${training_id}

python train.py --dataset LC --model deeplabv3+ --training_id ${training_id}
python train.py --dataset CN --model deeplabv3+ --training_id ${training_id}
python train.py --dataset FS --model deeplabv3+ --training_id ${training_id}
