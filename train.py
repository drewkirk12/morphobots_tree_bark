import argparse
import os
import json
import hyperparameters as hp
from dataset.generate_dataset import GenerateDataset
from Trainer import Trainer


def load_config(config_path):
    config = open(config_path, 'r')
    config_args = {}
    for line in config:
        if line.find(' = ') != -1:
            name, value = line.split(' = ')
            config_args[name] = value.strip('\n')
    config.close()
    return config_args


def get_dataset_generator():
    dataset_gen = GenerateDataset(hp.DATASET_PATH)
    dataset_gen.load_dataset(hp.EXISTING_DATASET, float(hp.TRAIN_SPLIT))

    if hp.INCLUDE_IGNORE == 'True':
        dataset_gen.add_ignore()

    return dataset_gen

if __name__ == '__main__':

    dataset_generator = get_dataset_generator()

    use_k_fold = int(hp.K_FOLD) > 1

    if use_k_fold:
        dataset = dataset_generator.get_k_fold_dataset(int(hp.K_FOLD))
    else:
        dataset = dataset_generator.all_dataset(train_size=float(hp.TRAIN_SIZE), tree_size=float(hp.TREE_SIZE))
        dataset = {
            0: dataset
        }

    for i in range(int(hp.K_FOLD)):
        model_name = str(i)
        
        trainer = Trainer()

        trainer.train(n_epoch=int(hp.N_EPOCHS), folder=dataset[i], model_name=hp.MODEL,
                      print_info=hp.PRINT == 'TRUE')

        trainer.save_train_data()
