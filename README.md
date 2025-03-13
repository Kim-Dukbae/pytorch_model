## **pytorch model**

torch.__version__     ->  2.5.1 <br>


### ```colab```
```
!git clone https://github.com/Kimduckba/pytorch_model
!pip install torchsummary

import sys
sys.path.append("/content/pytorch_model")

```

| model  | parametes | input shape | review |
| lenet1 |  3,246    | (1, 28, 28) | conv(kernel=5) -> avg pool ->  conv(kernel=5) -> avg pool -> FC  |
| lenet5 |  28,886   | (1, 32, 32) | conv(kernel=5) -> avg pool ->  conv(kernel=5) -> avg pool -> FC -> FC |
| lenet  |  29,948   | (1, 32, 32) | conv(kernel=3)*2 -> max pool ->  conv(kernel=3)*2 -> max pool -> FC -> FC |


