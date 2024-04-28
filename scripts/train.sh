
training_id="zhr1"

python src/train.py --dataset LC --model pan --training_id ${training_id}
python src/train.py --dataset CN --model pan --training_id ${training_id}
python src/train.py --dataset FS --model pan --training_id ${training_id}
