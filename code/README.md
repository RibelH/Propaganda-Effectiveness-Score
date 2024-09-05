# Fine Grained Propaganda Detection
We assume that you will be running training/evaluation on your local machine which has multiple GPUs like GTX1080.

1. pip install -r requirements.txt 
2. For the training, Run one of the following: 
```python train.py --bert --training --batch_size 16 --lr 3e-5 --n_epochs 20 --patience 7```
```python train.py --joint --training --batch_size 16 --lr 3e-5 --alpha 0.9 --n_epochs 20 --patience 7```
```python train.py --granu --training --batch_size 16 --lr 3e-5 --alpha 0.9 --n_epochs 20 --patience 7```
```python train.py --mgn --sig --training --batch_size 16 --lr 3e-5 --alpha 0.9 --n_epochs 20 --patience 7```

3. For the fragment-level evaluation, Run the evaluate.sh 
```./evaluate.sh ./results/[output.file] bert```
```./evaluate.sh ./results/[output.file] bert-joint```
```./evaluate.sh ./results/[output.file] bert-granu```
```./evaluate.sh ./results/[output.file] mgn```

4. For the span-level evaluation, Run the span-evaluate.sh 
```./span_evaluate.sh ./results/[output.file] bert```
```./span_evaluate.sh ./results/[output.file] bert-joint```
```./span_evaluate.sh ./results/[output.file] bert-granu```
```./span_evaluate.sh ./results/[output.file] mgn```

## Tested on:
Python 3.12 <br>
CUDA 9
Torch 1.0 <br>
huggingface/pytorch-pretrained-BERT **0.4 **<br>


