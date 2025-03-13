## **pytorch model**

torch.__version__     ->  2.5.1 <br>


### ```colab```
```
!git clone https://github.com/Kimduckba/pytorch_model
!pip install torchsummary

import sys
sys.path.append("/content/pytorch_model")

```
| <sub>Model  | <sub>Parameters | <sub>Input Shape | <sub>Review                                      |
|--------|------------|-------------|---------------------------------------------|
| <sub>LeNet1 | <sub>3,246      | <sub>(1, 28, 28) |<sub> convolution의 kernel은 5이며, activation fucntion은 tanh로 fc로 바로 직접 연결됨. |
| <sub>LeNet5 | <sub>28,886     | <sub>(1, 32, 32) |<sub> convolution의 kernel은 5이며, activation fucntion은 tanh로 fc에 hidden layer 하나 추가함. |
| <sub>LeNet  | <sub>29,948     | <sub>(1, 32, 32) |<sub> convolution의 kernel를 3으로 줄이며 연속 두 번 사용 후, activation fucntion은 relu로 변경함. |
|<sub> VGG16 |<sub>  ? |<sub> 변경 가능 |<sub> VGG16(in_channels= 3,  img_size= 224, num_classes= 1000) |
|<sub> 0 |<sub>  1|<sub>  2|<sub>  3|
