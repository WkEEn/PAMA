# PAMA:Position-aware masked autoencoder for histopathology WSI representation learning
This is a PyTorch implementation of the paper [PAMA](https://doi.org/10.1007/978-3-031-43987-2_69):



### Pre-train

Run the codes on the slurm with multiple GPUs:
```
#!/bin/bash

#SBATCH -w gpu0[1]
#SBATCH --gres=gpu:2
#SBATCH -N 1
#SBATCH -p com
#SBATCH --cpus-per-task=40
#SBATCH -o tcgaLung_pama_pretrain.log

srun python ./main_pretrain.py \
  --dist-url 'tcp://localhost:10001' \
  --b 18 \
  --train './data/train.csv' \
  --mask_ratio 0.75 \
  --in-chans 384 \
  --lr 1e-3 \
  --epochs 100 \
  --multiprocessing-distributed \
  --save-path ./checkpoints/tcgaLung_pama_pretrain \
  /data_path

```

### Fine-tuning and linear-probing

Run on on multiple GPUs:
```
#!/bin/bash

#SBATCH -w gpu0[1]
#SBATCH --gres=gpu:2
#SBATCH -N 1
#SBATCH -p com
#SBATCH --cpus-per-task=24
#SBATCH -o tcgaLung_pama_finetune.log

source activate my_base
srun python ./main_finetune.py \
  --dist-url 'tcp://localhost:10001' \
  --b 12 \
  --train './data/train.csv' \
  --test './data/test.csv' \
  --finetune "./checkpoints/multi_organ_pretrain.pth.tar" \
  --in-chans 384 \
  --lr 1e-3 \
  --epochs 30 \
  --num-classes 3 \
  --weighted-sample \
  --multiprocessing-distributed \
  --save-path ./tcgaLung_pama_finetune/ \
  /data_path
```

```
#!/bin/bash

#SBATCH -w gpu0[1]
#SBATCH --gres=gpu:2
#SBATCH -N 1
#SBATCH -p com
#SBATCH --cpus-per-task=24
#SBATCH -o tcgaLung_pama_linear.log

source activate my_base
srun python ./main_linprobe.py \
  --dist-url 'tcp://localhost:10001' \
  --b 12 \
  --train './data/train.csv' \
  --test './data/test.csv' \
  --finetune "./checkpoints/multi_organ_pretrain.pth.tar" \
  --in-chans 384 \
  --lr 1e-3 \
  --epochs 30 \
  --num-classes 3 \
  --weighted-sample \
  --multiprocessing-distributed \
  --save-path ./tcgaLung_pama_linear/ \
  /data_path
```



If the code is helpful to your research, please cite:
```
@InProceedings{10.1007/978-3-031-43987-2_69,
author="Wu, Kun 
and Zheng, Yushan
and Shi, Jun
and Xie, Fengying
and Jiang, Zhiguo",
title="Position-Aware Masked Autoencoder for Histopathology WSI Representation Learning",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="714--724",
isbn="978-3-031-43987-2"
}
```