from torch import nn as nn, __init__


class LinearBlockUnit(nn.Module):
    def __init__(self, input_size, output_size, stride=1):
        super(LinearBlockUnit, self).__init__()
        self.act = nn.ReLU()
        #self.conv = nn.Conv2d(1, 10, kernel_size=3, stride=stride, padding=1)
        self.linear = nn.Linear(input_size, output_size)
        self.bn = nn.BatchNorm1d(output_size)

    def forward(self, x):
        out = self.act(x)
        out = self.linear(out)
        out = self.bn(out)
        return out


class FiveWaySparseBlockNet(nn.Module):
    def __init__(self, input_size, output_classes, hidden_block_size=100):
        super(FiveWaySparseBlockNet, self).__init__()

        self.input_blocks = nn.ModuleList()
        for _ in range(5):
            self.input_blocks.append(LinearBlockUnit(input_size, hidden_block_size))

        self.hidden_blocks = nn.ModuleList()
        for _ in range(5):
            self.hidden_blocks.append(LinearBlockUnit(hidden_block_size, hidden_block_size))

        self.output_blocks = nn.ModuleList()
        for _ in range(5):
            self.output_blocks.append(LinearBlockUnit(hidden_block_size, output_classes))

    def forward(self, x):
        input_to_blocks = {}
        for idx, block_unit in enumerate(self.input_blocks):
            input_to_blocks[idx] = block_unit(x)

        hidden_block_results = {}
        for idx, block_unit in enumerate(self.hidden_blocks):
            hidden_block_results[idx] = block_unit(input_to_blocks[idx])

        output_blocks = []
        for idx, block_unit in enumerate(self.output_blocks):
            output_blocks.append(block_unit(hidden_block_results[idx]))

        return torch.mean(torch.stack(output_blocks), dim=0)