import sys

import torch
from torch.utils import data
import os
from hp import hp
from model import BertMultiTaskLearning
from data_load import PropDataset, pad, idx2tag, num_task
import time
from collections import OrderedDict


timestr = time.strftime("%Y%m%d-%H%M%S")


def propgen_eval(model, iterator, f):
    model.eval()

    Words, Is_heads = [], []
    Tags = [[] for _ in range(num_task)]
    Y_hats = [[] for _ in range(num_task)]
    with torch.no_grad():

        for _, batch in enumerate(iterator):
            words, x, is_heads, att_mask, tags, y, seqlens = batch
            att_mask = torch.Tensor(att_mask)
            logits, y_hats = model(x, attention_mask=att_mask)  # logits: (N, T, VOCAB), y: (N, T)

            for i in range(num_task):
                logits[i] = logits[i].view(-1, logits[i].shape[-1])  # (N*T, 2)

            Words.extend(words)
            Is_heads.extend(is_heads)

            for i in range(num_task):
                Tags[i].extend(tags[i])
                Y_hats[i].extend(y_hats[i].cpu().numpy().tolist())

    with open(f, 'w', encoding='utf-8') as fout:
        y_hats, preds = [[] for _ in range(num_task)], [[] for _ in range(num_task)]

        for words, is_heads, tags[0], tags[1], y_hats[0], y_hats[1] in zip(Words, Is_heads, *Tags, *Y_hats):
            y_hats[0] = [hat for head, hat in zip(is_heads, y_hats[0]) if head == 1]
            preds[0] = [idx2tag[0][hat] for hat in y_hats[0]]
            preds[1] = idx2tag[1][y_hats[1]]

            fout.write(words.split()[0])
            fout.write("\n")
            for w, t1, p_1 in zip(words.split()[2:-1], tags[0].split()[1:-1], preds[0][1:-1]):
                fout.write("{} {} {}\n".format(w, p_1, preds[1]))
            fout.write("\n")

if __name__=="__main__":
    model = BertMultiTaskLearning.from_pretrained('bert-base-cased')

    result_path = os.path.join('results', timestr)
    evaluated_model = hp.evaluated_model

    state_dict = torch.load(os.path.join('checkpoints', hp.checkpoint))

    # Fixing loading issues
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' in k:
            k = k.replace('module.', '')
        else:
            k = k
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    propgen_dataset = PropDataset(hp.propgenset.format(evaluated_model), True)
    propgen_iter = data.DataLoader(dataset=propgen_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=pad)

    propgen_eval(model, propgen_iter, result_path)
    sys.exit()