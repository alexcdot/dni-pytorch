import torch
import torch.nn as nn
import torch.optim as optim
import dni
import numpy as np
from torchsummary import summary
import matplotlib
from matplotlib import pyplot as plt
import tkinter
import torch.autograd.profiler as profiler
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


class convs(nn.Module):
    def __init__(self):
        super(convs, self).__init__()
        self.input_trigger = nn.Conv2d(512, 512, kernel_size=5, padding=2).to('cuda:0')
        self.bn = nn.BatchNorm2d(512).to('cuda:0')
        self.input_context = nn.Linear(10, 16)
        self.hidden = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        self.output = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        # zero-initialize the last layer, as in the paper
        nn.init.constant(self.input_trigger.weight, 0)
        # self.context = None

    def forward(self, x):
        x = self.input_trigger(x)
        x = self.bn(x)
        # x = self.hidden(x)
        # x = self.output(x)
        # context = None
        return x


class convs1(nn.Module):
    def __init__(self):
        super(convs1, self).__init__()
        self.input_trigger = nn.Conv2d(512, 512, kernel_size=3, padding=1).to('cuda:0')
        self.bn = nn.BatchNorm2d(512).to('cuda:0')
        self.input_context = nn.Linear(10, 16)
        self.hidden = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        self.output = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        # zero-initialize the last layer, as in the paper
        nn.init.constant(self.input_trigger.weight, 0)
        # self.context = None

    def forward(self, x):
        x = self.input_trigger(x)
        x = self.bn(x)
        # x = self.hidden(x)
        # x = self.output(x)
        # context = None
        return x


class ConvSynthesizer(nn.Module):
    def __init__(self):
        super(ConvSynthesizer, self).__init__()
        self.input_trigger = nn.Conv2d(512, 512, kernel_size=5, padding=2).to('cuda:0')  #
        self.bn = nn.BatchNorm2d(512).to('cuda:0')                          #
        self.input_context = nn.Linear(10, 16)
        self.hidden = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        self.output = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        # zero-initialize the last layer, as in the paper
        nn.init.constant(self.input_trigger.weight, 0)
        # self.context = None

    def forward(self, trigger, context):
        x = self.input_trigger(trigger.to('cuda:0'))    #modified here
        x = self.bn(x)
        # x = self.hidden(x)
        # x = self.output(x)
        # context = None
        if context is not None:
            x += (
                self.input_context(context).unsqueeze(2)
                    .unsqueeze(3)
                    .expand_as(x)
            )

        return x


class Conv_less(nn.Module):
    def __init__(self):
        super(Conv_less, self).__init__()
        self.input_trigger = nn.Conv2d(512, 512, kernel_size=3, padding=1).to('cuda:0')
        self.bn = nn.BatchNorm2d(512).to('cuda:0')
        self.input_context = nn.Linear(10, 16)
        self.hidden = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        self.output = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        # zero-initialize the last layer, as in the paper
        nn.init.constant(self.input_trigger.weight, 0)
        # self.context = None

    def forward(self, trigger, context):
        x = self.input_trigger(trigger)
        x = self.bn(x)
        # x = self.hidden(x)
        # x = self.output(x)
        # context = None
        if context is not None:
            x += (
                self.input_context(context).unsqueeze(2)
                    .unsqueeze(3)
                    .expand_as(x)
            )

        return x


class front(ResNet):
    def __init__(self, *args, **kwargs):
        super(front, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cuda:0')  # changed from cpu to gpu

    def forward(self, x):
        x = self.seq1(x)
        return x.view(x.size(0), -1)


class back(ResNet):
    def __init__(self, *args, **kwargs):
        super(back, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:0')

        self.fc.to('cuda:0')

    def forward(self, x):
        x = self.seq2(x.to('cuda:0'))

        return self.fc(x.view(x.size(0), -1))


class front_dni(ResNet):
    def __init__(self, *args, **kwargs):
        super(front_dni, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cuda:0')  # changed from cpu to gpu

        # self.input_context = nn.Linear(10, 16)

        # self.input_trigger = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        # self.hidden = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        # self.output = nn.Conv2d(512, 512, kernel_size=5, padding=2)

        context_dim = None
        self.backward_interface = dni.BackwardInterface(ConvSynthesizer())

        # self.backward_interface = dni.BackwardInterface( dni.BasicSynthesizer(mcf()))

    def forward(self, x):
        x = self.seq1(x)
        # print(x.shape)
        context = None
        with dni.synthesizer_context(context):
            x = self.backward_interface(x)

        return x.view(x.size(0), -1)


class flat(ResNet):
    def __init__(self, *args, **kwargs):
        super(flat, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cuda:0')  # changed from cpu to gpu

        # self.input_context = nn.Linear(10, 16)

        # self.input_trigger = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        # self.hidden = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        # self.output = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        self.input_trigger = nn.Conv2d(512, 512, kernel_size=5, padding=2).to('cuda:0')
        self.bn = nn.BatchNorm2d(512).to('cuda:0')

        # self.backward_interface = dni.BackwardInterface( dni.BasicSynthesizer(mcf()))

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:0')

        self.fc.to('cuda:0')

    def forward(self, x):
        x = self.seq1(x)
        # print(x.shape)
        x = self.input_trigger(x)
        x = self.bn(x)

        # x = self.input_trigger(x)
        # x = self.hidden(x)
        # x = self.output(x)

        x = self.seq2(x.to('cuda:0'))

        return self.fc(x.view(x.size(0), -1))


class less(ResNet):
    def __init__(self, *args, **kwargs):
        super(less, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cuda:0')  # changed from cpu to gpu

        # self.input_context = nn.Linear(10, 16)

        # self.input_trigger = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        # self.hidden = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        # self.output = nn.Conv2d(512, 512, kernel_size=5, padding=2)

        context_dim = None
        self.backward_interface = dni.BackwardInterface(Conv_less())

        #self.backward_interface = dni.BackwardInterface( dni.BasicSynthesizer(mcf()))

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:0')

        self.fc.to('cuda:0')

    def forward(self, x):
        x = self.seq1(x)
        # print(x.shape)
        context = None
        with dni.synthesizer_context(context):
            x = self.backward_interface(x)

        # x = self.input_trigger(x)
        # x = self.hidden(x)
        # x = self.output(x)

        x = self.seq2(x.to('cuda:0'))

        return self.fc(x.view(x.size(0), -1))


model = convs()
summary(model, (512, 16, 16))

model = convs1()
summary(model, (512, 16, 16))

model = front()
summary(model, (3, 128, 128))

model = back()
summary(model, (512, 16, 16))


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
        ).to('cuda:0')  # changed from cpu to gpu

        # self.input_context = nn.Linear(10, 16)

        # self.input_trigger = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        # self.hidden = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        # self.output = nn.Conv2d(512, 512, kernel_size=5, padding=2)

        context_dim = None
        #self.backward_interface = dni.BackwardInterface(ConvSynthesizer()).to('cuda:0')  #dual to

        # self.backward_interface = dni.BackwardInterface( dni.BasicSynthesizer(mcf()))

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:1')

        self.fc.to('cuda:1')

    def forward(self, x):
        x = self.seq1(x.to('cuda:0'))
        # print(x.shape)
        context = None
        #with dni.synthesizer_context(context):
            #print("enter")
            #x = self.backward_interface(x)
            #print("out1")
            #x = self.seq2(x.to('cuda:1'))


        # x = self.input_trigger(x1)
        # x = self.hidden(x)
        # x = self.output(x)

        print("enter2")

        x = self.seq2(x.to('cuda:1'))

        return self.fc(x.view(x.size(0), -1))





class PipelineParallelResNet49(ModelParallelResNet50):
    def __init__(self, split_size=30, *args, **kwargs):
        super(PipelineParallelResNet49, self).__init__(*args, **kwargs)
        self.split_size = split_size

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        s_prev = self.seq1(s_next).to('cuda:1')
        ret = []

        for s_next in splits:
            # A. s_prev runs on cuda:1
            s_prev = self.seq2(s_prev)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

            # B. s_next runs on cuda:0, which can run concurrently with A
            s_prev = self.seq1(s_next).to('cuda:1')

        s_prev = self.seq2(s_prev)
        ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

        return torch.cat(ret)







class PipelineParallelResNet50(ModelParallelResNet50):
    def __init__(self, split_size=30, *args, **kwargs):
        super(PipelineParallelResNet50, self).__init__(*args, **kwargs)
        self.split_size = split_size

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        x = self.seq1(s_next).to('cuda:0')
        context = None
        with dni.synthesizer_context(context):
            # print("enter")
            s_prev = self.backward_interface(x.to('cuda:0')).to('cuda:1')

        ret = []

        for s_next in splits:
            # A. s_prev runs on cuda:1
            s_prev = self.seq2(s_prev)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

            # B. s_next runs on cuda:0, which can run concurrently with A
            x = self.seq1(s_next).to('cuda:0')
            context = None
            with dni.synthesizer_context(context):
                # print("enter")
                s_prev = self.backward_interface(x.to('cuda:0')).to('cuda:1')

        s_prev = self.seq2(s_prev)
        ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))

        return torch.cat(ret)

















import torchvision.models as models

num_batches = 5
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
        outputs = model(inputs.to('cuda:0'))  # be careful here cpu
        # print('outputs', outputs)

        # run backward pass
        labels = labels.to(outputs.device)
        # print('labels',labels)

        loss = loss_fn(outputs, labels)
        loss.backward()

        # print('loss', loss)
        # loss_av += loss
        if len(mf_network) + len(ori_network) < 2 * num_batches:
            if len(mf_network) > (num_batches - 1):
                ori_network.append(loss)
            else:
                # print('loss', loss)
                mf_network.append(loss)

        optimizer.step()

    # loss_av = loss_av/num_batches


import matplotlib.pyplot as plt
# plt.switch_backend('Agg')
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
# model = ModelParallelResNet50()
# summary(model, (3, 128, 128))




model = ModelParallelResNet50()

with profiler.profile() as prof:
    with profiler.record_function("model_train"):
        train(model)


prof.export_chrome_trace("trace_original.json")









setup = "model = ModelParallelResNet50()"
# globals arg is only available in Python 3. In Python 2, use the following
# import __builtin__
# __builtin__.__dict__.update(locals())
mp_run_times = timeit.repeat(
    stmt, setup, number=1, repeat=num_repeat, globals=globals())
mp_mean, mp_std = np.mean(mp_run_times), np.std(mp_run_times)

setup = "import torchvision.models as models;" + \
        "model = models.resnet50(num_classes=num_classes).to('cuda:0')"
rn_run_times = timeit.repeat(
    stmt, setup, number=1, repeat=num_repeat, globals=globals())
rn_mean, rn_std = np.mean(rn_run_times), np.std(rn_run_times)

setup = "model = flat()"
# globals arg is only available in Python 3. In Python 2, use the following
# import __builtin__
# __builtin__.__dict__.update(locals())
fr_run_times = timeit.repeat(
    stmt, setup, number=1, repeat=num_repeat, globals=globals())
fr_mean, fr_std = np.mean(fr_run_times), np.std(fr_run_times)

setup = "model = less()"
# globals arg is only available in Python 3. In Python 2, use the following
# import __builtin__
# __builtin__.__dict__.update(locals())
le_run_times = timeit.repeat(
    stmt, setup, number=1, repeat=num_repeat, globals=globals())
le_mean, le_std = np.mean(le_run_times), np.std(le_run_times)

# plt.switch_backend('TkAgg')


plt.style.use('classic')
fig, ax = plt.subplots()
a = np.linspace(1, num_batches, num_batches)
plt.plot(a, mf_network, 'b', label='DNI integrated')
plt.plot(a, ori_network, 'g', label='original')
plt.title('loss chart compare')
plt.legend()
plt.show()


# plt.plot(a, ori_network)
# plt.title('original')
# plt.show()




#model = ModelParallelResNet50()
model = PipelineParallelResNet49()

with profiler.profile() as prof:
    with profiler.record_function("model_train"):
        train(model)


prof.export_chrome_trace("trace_pipe.json")




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


plot([mp_mean, rn_mean, fr_mean, le_mean],
     [mp_std, rn_std, fr_std, le_std],
     ['DNI model', 'GPU Original', 'GPU extended', 'DNI less'],
     'mp_vs_rn_vs_fr_vs_fdn.png')