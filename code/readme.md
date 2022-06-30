# RobustLR: A Diagnostic Benchmark for Evaluating Logical Robustness of Deductive Reasoners

![image-20220624212232784](https://tva1.sinaimg.cn/large/e6c9d24egy1h3kdgjc0g7j20l80kqtam.jpg)

Above figures show an examples of deductive reasoning tasks that we want to test on the current popular model i.e RoBerta, T5-Large, T5-3B, T5-11B ,and check the robustness.



### Dependencies

+ Dependencies can be installed using `requirements.txt`.

### Datasets for Training and Evaluation
Please download the data using [this link](https://drive.google.com/file/d/1xq92DOyH_eS37L5SyOzu3Vo_4RSiNsm0/view?usp=sharing) and place it inside the `data` folder.

### Pipeline for running code

+ Use the `commands.sh`

1.You first need to generate corresponding tokenizations based on your dataset and model

2.Then need to train a model based and saved the checkpoints at `../saved/`

3.Evaluation on 3 different types of contrast sets and 3 different types of equivalent sets from the previous saved checkpoint.

```
#!/bin/bash

# Use all operators at train time [AND, OR, NOT]

# process data
python process_dataset.py --dataset logic_comp1_v0 --dataset_type 1.0_balance_new3_seed42 --filtered --trim

# train roberta checkpoint
python main.py --dataset logic_comp1_v0_1.0_balance_new3_seed42_filtered --train_dataset logic_comp1_v0_1.0_balance_new3_seed42_filtered --dev_dataset logic_comp1_v0_1.0_balance_new3_seed42_filtered --test_dataset logic_comp1_v0_1.0_balance_new3_seed42_filtered


# Evaluation on RobustLR diagnostic benchmark

# Conjunction Contrast Set
python process_dataset.py --dataset logic_comp1_v7 --dataset_type 1.0_new3_seed42 --filtered --trim --eval
python main.py --override evaluate --dataset logic_comp1_v7_1.0_new3_seed42_filtered --train_dataset logic_comp1_v7_1.0_new3_seed42_filtered --dev_dataset logic_comp1_v7_1.0_new3_seed42_filtered --test_dataset logic_comp1_v7_1.0_new3_seed42_filtered --ckpt_path <model_ckpt>

# Disjunction Contrast Set
python process_dataset.py --dataset logic_comp1_v9 --dataset_type 1.0_new3_seed42 --filtered --trim --eval
python main.py --override evaluate --dataset logic_comp1_v9_1.0_new3_seed42_filtered --train_dataset logic_comp1_v9_1.0_new3_seed42_filtered --dev_dataset logic_comp1_v9_1.0_new3_seed42_filtered --test_dataset logic_comp1_v9_1.0_new3_seed42_filtered --ckpt_path <model_ckpt>

# Negation Contrast Set
python process_dataset.py --dataset logic_comp1_v14 --dataset_type 1.0_new3_seed42 --filtered --trim --eval
python main.py --override evaluate --dataset logic_comp1_v14_1.0_new3_seed42_filtered --train_dataset logic_comp1_v14_1.0_new3_seed42_filtered --dev_dataset logic_comp1_v14_1.0_new3_seed42_filtered --test_dataset logic_comp1_v14_1.0_new3_seed42_filtered --ckpt_path <model_ckpt>

# Contrapositive Equivalence Set
python process_dataset.py --dataset logic_equiv_eq_v0 --dataset_type 1.0_balance_new3_seed42 --filtered --trim --eval
python main.py --override evaluate --dataset logic_equiv_eq_v0_1.0_balance_new3_seed42_filtered --train_dataset logic_equiv_eq_v0_1.0_balance_new3_seed42_filtered --dev_dataset logic_equiv_eq_v0_1.0_balance_new3_seed42_filtered --test_dataset logic_equiv_eq_v0_1.0_balance_new3_seed42_filtered --ckpt_path <model_ckpt>

# Distributive 1 Equivalence Set
python process_dataset.py --dataset logic_equiv2_eq_v0 --dataset_type 1.0_balance_new3_seed42 --filtered --trim --eval
python main.py --override evaluate --dataset logic_equiv2_eq_v0_1.0_balance_new3_seed42_filtered --train_dataset logic_equiv2_eq_v0_1.0_balance_new3_seed42_filtered --dev_dataset logic_equiv2_eq_v0_1.0_balance_new3_seed42_filtered --test_dataset logic_equiv2_eq_v0_1.0_balance_new3_seed42_filtered --ckpt_path <model_ckpt>

# Distributive 2 Equivalence Set
python process_dataset.py --dataset logic_equiv3_eq_v0 --dataset_type 1.0_balance_new3_seed42 --filtered --trim --eval
python main.py --override evaluate --dataset logic_equiv3_eq_v0_1.0_balance_new3_seed42_filtered --train_dataset logic_equiv3_eq_v0_1.0_balance_new3_seed42_filtered --dev_dataset logic_equiv3_eq_v0_1.0_balance_new3_seed42_filtered --test_dataset logic_equiv3_eq_v0_1.0_balance_new3_seed42_filtered --ckpt_path <model_ckpt>

```
