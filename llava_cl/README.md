# Install

```
git clone CLMM
cd CLMM

conda create -n clmm python=3.10
conda activate clmm
pip install --upgrade pip
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -e .

pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

- Upgrade gcc

```
tar xzf gcc-11.4.0.tar.gz
cd gcc-11.4.0
./contrib/download_prerequisites
cd ..
mkdir GCC-11.4.0
cd GCC-11.4.0
$PWD/../gcc-11.4.0/configure --prefix=$HOME/GCC-11.4.0 --enable-languages=c,c++,fortran,go
make -j 64
make install
```

# Model Preparation

For offline compute node, edit `.env` file like:

```
BERT_BASE_UNCASED=/path/to/bert-base-uncased
INSTRUCTBLIP_VICUNA_7B=/path/to/instructblip-vicuna-7b
LLAVA_V1_5_7B=/path/to/llava-v1.5-7b
```

# Data Preparation

Download datasets and put it under `/playground/data`.
Download `split_data.zip` for train data.
Download `blip_train_data.zip` for train data.
Download `eval.zip` for evaluation.

Extract all files and organize the directory like:

```
blip_train_data
split_data
eval

coco
    - train2017
    - train2014
    - val2017
    - val2014
    - test2015
flickr30k
    - flickr30k-images
gqa
    - images
ocr_vqa
    - images
vizwiz
    - images
textvqa
    - train_images
    - test_images
```


# Train

```
bash scripts/instruct_blip/finetune_task_cl.sh order1 tir+mas 1e2

bash scripts/llava_v1_5/finetune_task_cl.sh order1 tir+mas 1e2
```

# TO-DO List
 - [ ] Support CL method DER
 - [ ] Support CL method L2P
 - [ ] Replay + Task-similarity-informed strategy
