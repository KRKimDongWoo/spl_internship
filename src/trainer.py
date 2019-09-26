import torch.nn as nn
from torch import Tensor
import torch
from .builder import ModelBuilder

def progress(loader):
    batch_num = (len(loader.dataset) + 1) // loader.batch_size
    return batch_num // 20

class ModelTrainer():
    def __init__(self, max_epochs, patient, 
                 criterion, optimizer, 
                 train_loader, test_loader, 
                 device='cpu'):
        
        self.builder = ModelBuilder()
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_progress = progress(self.train_loader)
        self.test_progress = progress(self.test_loader)
        
        self.max_epochs = max_epochs
        self.patient = patient
        self.criterion = criterion()
        self.opt_class = optimizer

        self.device = torch.device(device)

    def build(self, graph):
        return self.builder.set_graph(graph).build()

    def train(self, net):
        train_loss, total = 0., 0
        print('  - Train : ', end='')
        for i, data in enumerate(self.train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = net(inputs)                
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            total += labels.size(0)
            
            if i % self.train_progress == self.train_progress - 1:
                print('*', end='')

        print(' train_loss: %.3f' % (train_loss/total))
    
    def test(self, net):
        correct, total = 0, 0
        print('  - Test  : ', end='')
        with torch.no_grad():
            for i, data in enumerate(self.test_loader, 0):
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                if i % self.test_progress == self.test_progress - 1:
                    print('*', end='')
        print(' accuracy: %.3f%%' % (correct/total*100))

        return 100 * correct / total

    def fit(self, graph):
        net = self.builder.set_graph(graph).build()
        net.to(self.device)
        net = nn.DataParallel(net)
        self.optimizer = self.opt_class(net.parameters(), lr=1e-3, momentum=0.9)

        max_accuracy = 0
        patient = 0
        fianl_epoch = 0

        for epoch in range(self.max_epochs):
            print(f' > epoch {epoch}/{self.max_epochs}')
            self.train(net)
            accuracy = self.test(net)

            if max_accuracy <= accuracy:
                patient += 1
            else:
                max_accuracy = accuracy
                patient = 0
            
            if patient >= self.patient:
                final_epoch = epoch
                break;

        return graph, max_accuracy, final_epoch
