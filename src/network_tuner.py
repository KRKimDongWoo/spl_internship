from .builder import ModelBuilder
from .trainer import ModelTrainer
from .generator import Generator
from .storage import Storage

import time

from queue import PriorityQueue

class NetworkTuner():
    def __init__(self, train_loader, test_loader, path='./'):
        self.train_loader = train_loader
        self.test_loader = test_loader

        in_shape, out_shape = self.verify_data_shape()

        self.graphs = PriorityQueue()
        self.untrained_graphs = []
        
        self.storage = Storage(path)
        self.trainer = None
        self.generator = Generator(*self.verify_data_shape(), self.storage)

        self.best_accuracy = 0
        self.best_model = 0

    def set_train_options(self, max_epochs, patient, 
                          criterion, optimizer, 
                          device='cpu'):
        self.trainer = ModelTrainer(max_epochs, patient,
                                    criterion, optimizer,
                                    self.train_loader, self.test_loader,
                                    device)
        return self.trainer

    def verify_data_shape(self):
        in_shape = self.train_loader.dataset.data[0].shape
        in_shape = in_shape[-1:] + in_shape[:-1]
        out_shape = (len(self.train_loader.dataset.classes),)
        return in_shape, out_shape

    def tune_until(self, timeout):
        start_time = time.time()
        print('Start tunning....')
        elapsed = 0
        while elapsed < timeout:
            if self.graphs.qsize() == 0:
                print('making base model')
                base = self.generator.generate()
                self.untrained_graphs.append((-1,) + base)
            else:
                for _ in range(min(5, self.graphs.qsize())):
                    parent = self.graphs.get()
                    print('generate from parent')
                    for __ in range(2):
                        gen_info = self.generator.generate(best)
                        self.untrained_graphs.append((best,) + gen_info)
                    self.untrained_graphs.append((-1, parent, []))

            for parent, model_num, history in self.untrained_graphs:
                if time.time() - start_time >= timeout:
                    elapsed = time.time() - start_time
                    break;
                
                print('-'*40)
                print(f' model number: {model_num}')
                if parent >= 0:
                    print(f' parent model: {parent}')
                print(' history:')
                for h in history:
                    print(f'  - {h}')
                
                graph = self.storage.load(model_num)
                graph, acc, epoch = self.trainer.fit(graph)
                print(f' Stopped on epoch {epoch}')
                print(f' Best accuracy: {acc}%')
                print(' Saving weights...', end='')
                storage.save(graph, model_num)
                print(' Done!')
                self.graphs.put(acc, model_num)
                if self.best_accuracy < acc:
                    self.best_model = model_num
                    self.best_accuracy = acc
                print('-'*40)
            self.untrained_graphs = []

        return self.best_accuracy, self.best_model

    def tune_cycle(self, max_graph):
        return
