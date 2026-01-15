import jittor as jt
from jittor import nn
import numpy as np



# ShuffleNet Block
class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ShuffleNetUnit, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm(out_channels // 2)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=stride, padding=1,
                               groups=out_channels // 2, bias=False)
        self.bn2 = nn.BatchNorm(out_channels // 2)

        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm(out_channels)

    def execute(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.stride == 1:
            out = out + x
        return out


# ShuffleNet 模型
class ShuffleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ShuffleNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(24)
        self.relu = nn.ReLU()

        # Stage 1
        self.stage1 = nn.Sequential(
            ShuffleNetUnit(24, 240, stride=2),
            ShuffleNetUnit(240, 240, stride=1)
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            ShuffleNetUnit(240, 480, stride=2),
            ShuffleNetUnit(480, 480, stride=1)
        )

        # Stage 3
        self.stage3 = nn.Sequential(
            ShuffleNetUnit(480, 960, stride=2),
            ShuffleNetUnit(960, 960, stride=1)
        )

        # Global Average Pooling and Fully Connected Layer
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(960, num_classes)

    def execute(self, x):
        x = x.permute((0, 3, 1, 2)).float32()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)

        return x

    def predict(self, x, apply_softmax=False):
        with jt.no_grad():
            if isinstance(x, (list, tuple)):
                x = jt.array(x)
            elif isinstance(x, np.ndarray):
                x = jt.array(x)

            logits = self.execute(x)

            if apply_softmax:
                predictions = jt.nn.softmax(logits, dim=-1)
            else:
                predictions = logits
        return predictions

def shufflenet(num_classes=10):
    return ShuffleNet(num_classes)