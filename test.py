from torch.autograd import Variable
from torchvision.transforms import *
import torch
import os
import json
import math
from dataset.generate_dataset import GenerateDataset
from dataset.data_loader import get_loader
from torch.autograd import Variable
from PIL import Image
import numpy as np
import hyperparameters as hp
from model.model import Model

CROP_SIZE = 224
import time

class Test:

    def __init__(self, model_name, model_path=None, log_path=None, dataset=None, multitask=True):

        self.net = None
        self.test_file = None

        self.dataset = dataset
        self.classes = []

        #self._load_classes()
        #self.classes.sort()
        self._create_network()

    def run(self, test_file_name, folder):
        loader = self.get_loader(folder)
        data_loader = enumerate(loader)
        running_acc = 0
        total = 0
        for j in range(len(loader)):
            batch_input, targets = self.create_mini_batch(data_loader)
            output = self.net(batch_input)
            predictions = output.max(1)[1].type_as(targets)
            correct = predictions.eq(targets).sum()

            total += output.size(0)
            running_acc += correct.item()
        return running_acc/total

    def get_loader(self, folder):
        loader, _ = get_loader(folder['test']['files'], folder['test']['labels'], hp.BATCH_SIZE)
        return loader

    def create_mini_batch(self, batch_loader):
        batch = next(batch_loader)[1]
        return Variable(batch[0]), Variable(batch[1].type(torch.LongTensor))

    def run_single_crop(self, test_file_name, batch_size=32):
        self._create_test_file(test_file_name)
        batch_input = []
        batch_files = []
        test_size = len(self.dataset['files'])
        times = []
        for i, file in enumerate(self.dataset['files']):
            start = time.time()
            img = Image.open(file)
            crop = ToTensor()(RandomCrop(224)(img))

            batch_input.append(crop)
            batch_files.append(file)

            if (i+1) % batch_size == 0 or (i+1) == test_size:
                batch_input = Variable(torch.stack(batch_input), volatile=True)
                output = self.net(batch_input)

                if self.multitask:
                    predictions = output[0].max(1)[1].cpu().data.numpy().tolist()
                    dbh_predictions = output[1].cpu().data.numpy().tolist()
                    dbh_predictions = [item for sublist in dbh_predictions for item in sublist]
                else:
                    predictions = output.max(1)[1].cpu().data.numpy().tolist()
                    dbh_predictions = np.zeros(batch_size)

                for j, prediction in enumerate(predictions):
                    file = batch_files[j]
                    class_name = file.split('/')[-2]
                    #self.write_results(class_name, file, pred=prediction, dbh=dbh_predictions[j])

                batch_input = []
                batch_files = []
                end = time.time()
                times.append(end - start)
                print(sum(times) / len(times))

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
        self.net.load_state_dict(torch.load("saved_models/model_1.pth"))
        self.net.eval()

    def _create_test_file(self, test_file_name):
        self.test_file = open(self.log_path + self.model_path + '/{}'.format(test_file_name), 'w', 1)
        for i, target in enumerate(self.classes):
            self.test_file.write(target)
            if i != len(self.classes):
                self.test_file.write(', ')
        self.test_file.write('\n')

    @staticmethod
    def split_crops(img):
        crops = []
        for i in range(img.size[1] // CROP_SIZE):
            for j in range(img.size[0] // CROP_SIZE):
                start_y = i * CROP_SIZE
                start_x = j * CROP_SIZE

                crop = img.crop((start_x, start_y, start_x + CROP_SIZE, start_y + CROP_SIZE))
                crop = ToTensor()(crop)
                crops.append(crop)

        if len(crops) > 0:
            return torch.stack(crops)
        else:
            return []

    def get_class_predictions(self, output):
        if self.multitask:
            output = output[0]
        predictions = output.max(1)[1]
        predictions = predictions.cpu()

        predictions = predictions.data.numpy()
        flat_results = predictions.tolist()
        pred = max(set(flat_results), key=flat_results.count)
        return pred

    @staticmethod
    def get_dbh_predictions(output):
        dbh = torch.mean(output[1])
        dbh = dbh.data[0]
        return dbh

    def write_results(self, class_name, file_path, pred, dbh):
        self.test_file.write(
            '{}, {}, {}, {}\n'.format(file_path, self.classes.index(class_name),
                                      pred, dbh))
        print('{} - {}, {}'.format(self.classes.index(class_name), pred,
                                   math.fabs(int(file_path.split('/')[-1].split('_')[2]) / math.pi - dbh)))

def get_dataset_generator():
        dataset_gen = GenerateDataset(hp.DATASET_PATH)
        dataset_gen.load_dataset(hp.EXISTING_DATASET, float(hp.TRAIN_SPLIT))

        if hp.INCLUDE_IGNORE == 'True':
            dataset_gen.add_ignore()

        return dataset_gen


if __name__ == '__main__':
    model = str(0)
    
    
    
    dataset_file = os.path.join(hp.EXISTING_DATASET)
    dataset_file = open(dataset_file)
    loaded_dataset = json.load(dataset_file)
    loaded_dataset = loaded_dataset["0"]
    dataset_file.close()

    test = Test(model,  dataset=loaded_dataset, multitask=False)
    test_acc = test.run(test_file_name='test_run', folder = loaded_dataset)
    print("Test Accuracy: %d", test_acc * 100)
