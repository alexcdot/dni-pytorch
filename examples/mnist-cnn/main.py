import torch
import torch.nn as nn
import torch.optim as optim
import dni
import numpy as np
from torchsummary import summary
from matplotlib import pyplot as plt

import torch.nn.functional as F

from torchvision.models.resnet import ResNet, Bottleneck

num_classes = 1000

class mcf(ResNet):
    def __init__(self, *args, **kwargs):
        super(mcf, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cpu')

        self.fc.to('cpu')

    def forward(self, x):
        x = self.seq2(x.to('cpu'))

        return self.fc(x.view(x.size(0), -1))


class ConvSynthesizer(nn.Module):
    def __init__(self):
        super(ConvSynthesizer, self).__init__()
        self.input_trigger = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(512)
        self.input_context = nn.Linear(10, 16)
        self.hidden = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        self.output = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        # zero-initialize the last layer, as in the paper
        nn.init.constant(self.input_trigger.weight, 0)

    def forward(self, trigger, context):
        x = self.input_trigger(trigger)
        x = self.bn(x)
        #x = self.hidden(x)
        #x = self.output(x)
        if context is not None:
            x += (
                self.input_context(context).unsqueeze(2)
                                           .unsqueeze(3)
                                           .expand_as(x)
            )
        return x



class ModelParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cpu')

        # self.input_context = nn.Linear(10, 16)

        #self.input_trigger = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        #self.hidden = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        #self.output = nn.Conv2d(512, 512, kernel_size=5, padding=2)

        context_dim = None
        self.backward_interface = dni.BackwardInterface(ConvSynthesizer())

        #self.backward_interface = dni.BackwardInterface( dni.BasicSynthesizer(mcf()))

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:0')

        self.fc.to('cuda:0')

    def forward(self, x):
        x = self.seq1(x)
        #print(x.shape)
        context = None
        with dni.synthesizer_context(context):
          x = self.backward_interface(x)

        #x = self.input_trigger(x)
        #x = self.hidden(x)
        #x = self.output(x)

        x = self.seq2(x.to('cuda:0'))

        return self.fc(x.view(x.size(0), -1))

import torchvision.models as models

num_batches = 100
batch_size = 120
image_w = 128
image_h = 128
mf_network = []
ori_network = []

num_repeat = 1


def train(model):
    print('once')
    model.train(True)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    loss_av = 0

    one_hot_indices = torch.LongTensor(batch_size) \
                           .random_(0, num_classes) \
                           .view(batch_size, 1)

    for _ in range(num_batches):
        # generate random inputs and labels
        inputs = torch.randn(batch_size, 3, image_w, image_h)
        labels = torch.zeros(batch_size, num_classes) \
                      .scatter_(1, one_hot_indices, 1)

        # run forward pass
        optimizer.zero_grad()
        outputs = model(inputs.to('cpu'))
        #print('outputs', outputs)

        # run backward pass
        labels = labels.to(outputs.device)
        #print('labels',labels)

        loss = loss_fn(outputs, labels)
        loss.backward()

        #print('loss', loss)
        #loss_av += loss
        if len(mf_network) > (num_batches - 1):
            ori_network.append(loss)
        else:
            print('loss', loss)
            mf_network.append(loss)
        optimizer.step()


    #loss_av = loss_av/num_batches



import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np
import timeit




stmt = "train(model)"




class mcd(ResNet):
    def __init__(self, *args, **kwargs):
        super(mcd, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cuda:0')

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:0')

        self.fc.to('cuda:0')

    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:0'))
        return self.fc(x.view(x.size(0), -1))




model = mcd()
summary(model, (3, 128, 128))
print('seperation')
#model = ModelParallelResNet50()
#summary(model, (3, 128, 128))



setup = "model = ModelParallelResNet50()"
# globals arg is only available in Python 3. In Python 2, use the following
# import __builtin__
# __builtin__.__dict__.update(locals())
mp_run_times = timeit.repeat(
    stmt, setup, number=1, repeat=num_repeat, globals=globals())
mp_mean, mp_std = np.mean(mp_run_times), np.std(mp_run_times)

setup = "import torchvision.models as models;" + \
        "model = models.resnet50(num_classes=num_classes).to('cpu')"
rn_run_times = timeit.repeat(
    stmt, setup, number=1, repeat=num_repeat, globals=globals())
rn_mean, rn_std = np.mean(rn_run_times), np.std(rn_run_times)

plt.switch_backend('TkAgg')

a = np.linspace(1, num_batches, num_batches)
plt.plot(a, mf_network)
plt.title('mixed model')
plt.show()


plt.plot(a, ori_network)
plt.title('original')
plt.show()



def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('ResNet50 Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)


plot([mp_mean, rn_mean],
     [mp_std, rn_std],
     ['Model Parallel', 'Single GPU'],
     'mp_vs_rn.png')