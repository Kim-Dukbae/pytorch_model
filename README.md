# 🌟 PyTorch Model

**Torch Version**: `2.5.1`  
**Author**: [김덕배](https://github.com/Kimduckba)  
**Last Updated**: March 13, 2025

------
**안녕하세요.**

최대한 직관적으로 구현하려고 노력했습니다. 여러분들의 공부에 조금이나마 도움이 되었으면 하며, 오늘 하루도 열심히 공부합시다.

------

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.5.1 (`pip install torch==2.5.1`)

### Setup:

```bash
!git clone https://github.com/Kimduckba/pytorch_model
!pip install torchsummary

import sys
sys.path.append("/content/pytorch_model")

# Automatically select GPU if available
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

### 🛠️ Models Overview
| model  | 
| ------ |
| LeNet  | 
| VGG    | 
| ResNet | 
