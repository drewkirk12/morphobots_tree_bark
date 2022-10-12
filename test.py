from torch.autograd import Variable
from torchvision.transforms import *
import torch
import os
import json
from dataset.generate_dataset import GenerateDataset
from dataset.data_loader import get_loader
from torch.autograd import Variable
import hyperparameters as hp
from model.model import Model

CROP_SIZE = 224
import time

class Test:

    def __init__(self):
        self._create_network()

    def run(self, folder):
        loader = self.get_loader(folder)
        data_loader = enumerate(loader)
        running_acc = 0
        total = 0
        pred_list = []
        for j in range(len(loader)):
            batch_input, targets = self.create_mini_batch(data_loader)
            output = self.net(batch_input)
            predictions = output.max(1)[1]
            pred_list.append(predictions)
            if(hp.HAS_LABELS == True):
                correct = predictions.eq(targets).sum()
                total += output.size(0)
                running_acc += correct.item()
        if(hp.PREDICTION_PATH != None):
            with open(hp.PREDICTION_PATH, 'w') as filehandle:
                json.dump(pred_list.toList(), filehandle)
        return running_acc/total

    def get_loader(self, folder):
        loader, _ = get_loader(folder['test']['files'], folder['test']['labels'], hp.BATCH_SIZE)
        return loader

    def create_mini_batch(self, batch_loader):
        batch = next(batch_loader)[1]
        return Variable(batch[0]), Variable(batch[1].type(torch.LongTensor))

    def _get_specific_predictions(self, output, specific_classes):
        results = []
        class_index = []
        for class_name in specific_classes:
            class_index.append(self.classes.index(class_name))

        for i, crop in enumerate(output[0]):
            results.append([])
            for index in class_index:
                results[i].append(crop[index].data[0])

        preds =[]
        for result in results:
            preds.append(class_index[result.index(max(result))])

        return max(set(preds), key=preds.count)

    def _load_classes(self):
        for file in self.dataset['files']:
            class_name = file.split('/')[-2]
            if class_name not in self.classes:
                self.classes.append(class_name)

    def _create_network(self):
        self.net = Model(hp.MODEL, n_classes=hp.N_CLASSES)
        self.net.load_state_dict(torch.load(hp.OLD_MODEL))
        self.net.eval()

def get_dataset_generator():
        dataset_gen = GenerateDataset(hp.IMAGE_PATH)
        dataset_gen.load_dataset(hp.EXISTING_DATASET, float(1- hp.TEST_SPLIT))

        if hp.INCLUDE_IGNORE == 'True':
            dataset_gen.add_ignore()

        return dataset_gen


if __name__ == '__main__':
    test = Test()  

    dataset_generator = get_dataset_generator()

    dataset = dataset_generator.all_dataset(train_size=float(hp.TRAIN_SIZE), tree_size=float(hp.TREE_SIZE))
    dataset = {0: dataset}

    test_acc = test.run(folder = dataset[0])
    
    print("Test Accuracy: %d", test_acc * 100)
