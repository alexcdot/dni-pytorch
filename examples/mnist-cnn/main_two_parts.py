from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
from torchvision.models.resnet import ResNet, Bottleneck
import numpy as np
import time
import dni

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--dni-delay', type=int, default=0, metavar='N',
                    help='number of epochs to train before starting DNI (default: 0)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--dni', action='store_true', default=False,
                    help='enable DNI')
parser.add_argument('--context', action='store_true', default=False,
                    help='enable context (label conditioning) in DNI')
parser.add_argument('--use-resnet', action='store_true', default=False,
                    help='enable resnet instead of default CNN')
parser.add_argument('--save-space', action='store_true', default=False,
                    help='save space when using DNI by putting computed graph in CPU')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def one_hot(indexes, n_classes):
    result = torch.FloatTensor(indexes.size() + (n_classes,))
    if args.cuda:
        result = result.cuda()
    result.zero_()
    indexes_rank = len(indexes.size())
    result.scatter_(
        dim=indexes_rank,
        index=indexes.data.unsqueeze(dim=indexes_rank),
        value=1
    )
    return Variable(result)


class ResNetConvSynthesizer(nn.Module):
    def __init__(self):
        super(ResNetConvSynthesizer, self).__init__()
        num_filters = 256
        self.input_trigger = nn.Conv2d(num_filters, num_filters, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(num_filters)
        self.input_context = nn.Linear(10, num_filters)
        self.hidden = nn.Conv2d(num_filters, num_filters, kernel_size=5, padding=2)
        self.output = nn.Conv2d(num_filters, num_filters, kernel_size=5, padding=2)
        # zero-initialize the last layer, as in the paper
        nn.init.constant(self.input_trigger.weight, 0)

    def forward(self, trigger, context):
        x = self.input_trigger(trigger)
        x = self.bn(x)
        if context is not None:
            x += (
                self.input_context(context).unsqueeze(2)
                                           .unsqueeze(3)
                                           .expand_as(x)
            )
        x = self.hidden(F.relu(x))
        x = self.output(F.relu(x))
        return x


class ModelParallelResNet50(ResNet):
    def __init__(self, *nkwargs, **kwargs):
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=10, *nkwargs, **kwargs)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1
        )  #.to('cpu')

        if args.dni:
            self.backward_interface = dni.BackwardInterface(ResNetConvSynthesizer())

        self.seq2 = nn.Sequential(
            self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool,
        )  # .to('cuda:0')
        # self.fc.to('cuda:0')

    def forward(self, x, y=None, epoch=None, dni_delay=None):
        verbose = np.random.random() < 0.01

        if args.save_space:
            self.seq2.cpu()
            torch.cuda.empty_cache()

            if verbose:
                torch.cuda.reset_max_memory_allocated()

            self.seq1.cuda()
        
        x = self.seq1(x)

        if args.dni and self.training:
            if (epoch is None or dni_delay is None) or epoch > dni_delay:
                if args.context:
                    context = one_hot(y, 10)
                else:
                    context = None
                with dni.synthesizer_context(context):
                    x = self.backward_interface(x)
        
        if verbose:
            print("with seq 1, GPU mem:", torch.cuda.max_memory_allocated())

        if args.save_space:
            self.seq1.cpu()
            torch.cuda.empty_cache()

            if verbose:
                torch.cuda.reset_max_memory_allocated()

            self.seq2.cuda()
            
        x = self.seq2(x)  # .to('cuda:0'))
        
        if verbose:
            print("with seq 2, GPU mem:", torch.cuda.max_memory_allocated())

        x = self.fc(x.view(x.size(0), -1))    
        return F.log_softmax(x)



class NetConvPart(nn.Module):
    def __init__(self):
        super(NetConvPart, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv1_bn = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()

    def forward(self, x, y=None, epoch=None, dni_delay=None):
        x = F.relu(F.max_pool2d(self.conv1_bn(self.conv1(x)), 2))
        x = F.max_pool2d(self.conv2_drop(self.conv2_bn(self.conv2(x))), 2)
        return x


class NetDNIPart(nn.Module):
    def __init__(self):
        super(NetDNIPart, self).__init__()
        if args.dni:
            self.backward_interface = dni.BackwardInterface(ConvSynthesizer())

    def forward(self, x, y=None, epoch=None, dni_delay=None):
        if args.dni and self.training:
            if (epoch is None or dni_delay is None) or epoch > dni_delay:
                if args.context:
                    context = one_hot(y, 10)
                else:
                    context = None
                with dni.synthesizer_context(context):
                    x = self.backward_interface(x)
        return x
    
    
class NetFCPart(nn.Module):
    def __init__(self):
        super(NetFCPart, self).__init__()
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5, padding=2)
        self.conv3_bn = nn.BatchNorm2d(20)
        self.conv4 = nn.Conv2d(20, 20, kernel_size=5, padding=2)
        self.conv4_bn = nn.BatchNorm2d(20)
        self.fc1 = nn.Linear(320, 50)
        self.fc1_bn = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 10)
        self.fc2_bn = nn.BatchNorm1d(10)

    def forward(self, x, y=None, epoch=None, dni_delay=None):
        x = F.relu(x)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = x.view(-1, 320)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        x = self.fc2_bn(self.fc2(x))
        return F.log_softmax(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net_conv_part = NetConvPart()
        self.net_dni_part = NetDNIPart()
        self.net_fc_part = NetFCPart()
        self.net_fc_part.cpu()
        torch.cuda.empty_cache()

    def forward(self, x, y=None, epoch=None, dni_delay=None):
        """
        Try to only keep 2 of the 3 parts in GPU memory at time
        """
        verbose = np.random.random() < 0.01
        
        if args.save_space:
            self.net_fc_part.cpu()
            torch.cuda.empty_cache()

            if verbose:
                torch.cuda.reset_max_memory_allocated()

            self.net_conv_part.cuda()
        
        x = self.net_conv_part(x, y, epoch, dni_delay)
        x = self.net_dni_part(x, y, epoch, dni_delay)
        
        if verbose:
            print("with conv in, GPU mem:", torch.cuda.max_memory_allocated())

        if args.save_space:
            self.net_conv_part.cpu()
            torch.cuda.empty_cache()

            if verbose:
                torch.cuda.reset_max_memory_allocated()

            self.net_fc_part.cuda()
        
        x = self.net_fc_part(x, y, epoch, dni_delay)
        
        if verbose:
            print("with FC in, GPU mem:", torch.cuda.max_memory_allocated())
        return x

class ConvSynthesizer(nn.Module):
    def __init__(self):
        super(ConvSynthesizer, self).__init__()
        self.input_trigger = nn.Conv2d(20, 20, kernel_size=5, padding=2)
        self.input_context = nn.Linear(10, 20)
        self.hidden = nn.Conv2d(20, 20, kernel_size=5, padding=2)
        self.output = nn.Conv2d(20, 20, kernel_size=5, padding=2)
        # zero-initialize the last layer, as in the paper
        nn.init.constant(self.output.weight, 0)

    def forward(self, trigger, context):
        x = self.input_trigger(trigger)
        if context is not None:
            x += (
                self.input_context(context).unsqueeze(2)
                                           .unsqueeze(3)
                                           .expand_as(x)
            )
        x = self.hidden(F.relu(x))
        return self.output(F.relu(x))

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

if args.use_resnet:
    model = ModelParallelResNet50()
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root='../data', train=True,
            download=True, transform=transform),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=True, transform=transform), 
        batch_size=args.batch_size, shuffle=False, **kwargs)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
else:
    model = Net()
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)

def train(epoch, dni_delay=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data, target, epoch, args.dni_delay)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# import ipdb
# with ipdb.launch_ipdb_on_exception():


for epoch in range(1, args.epochs + 1):
    start_time = time.time()
    train(epoch)
    print("Train time for 1 epoch:", time.time() - start_time)
    start_time = time.time()
    test()
    print("Test time for 1 epoch:", time.time() - start_time)
