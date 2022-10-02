import argparse
import os
import json
import hyperparameters as hp
import torch
from dataset.generate_dataset import GenerateDataset
from Trainer import Trainer

def get_dataset_generator():
    dataset_gen = GenerateDataset(hp.IMAGE_PATH)
    dataset_gen.load_dataset(hp.EXISTING_DATASET, float(hp.TRAIN_SPLIT))

    if hp.INCLUDE_IGNORE == 'True':
        dataset_gen.add_ignore()

    return dataset_gen

if __name__ == '__main__':

    dataset_generator = get_dataset_generator()

    dataset = dataset_generator.all_dataset(train_size=float(hp.TRAIN_SIZE), tree_size=float(hp.TREE_SIZE))
    dataset = {0: dataset}

    model_name = str(0)
    
    trainer = Trainer()

    trainer.train(n_epoch=int(hp.N_EPOCHS), folder=dataset[0], model_name=hp.MODEL,
                    print_info = 'TRUE')

    trainer.save_train_data()
