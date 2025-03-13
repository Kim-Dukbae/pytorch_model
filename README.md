## **pytorch model**

torch.__version__     ->  2.5.1 <br>


### ```colab```
```
!git clone https://github.com/Kimduckba/pytorch_model
!pip install torchsummary

import sys
sys.path.append("/content/pytorch_model")

```

| Model  | Parameters | Input Shape | Review                                      |
|--------|------------|-------------|---------------------------------------------|
| LeNet1 | 3,246      | (1, 28, 28) | convolution의 kernel은 5이며, activation fucntion은 tanh로 fc로 바로 직접 연결됨. |
| LeNet5 | 28,886     | (1, 32, 32) | convolution의 kernel은 5이며, activation fucntion은 tanh로 fc에 hidden layer 하나 추가함. |
| LeNet  | 29,948     | (1, 32, 32) | convolution의 kernel를 3으로 줄이며 연속 두 번 사용 후, activation fucntion은 relu로 변경함. |
