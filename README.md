# MedTrans
Official PyTorch implementation of MedTrans  
## Dataset  
1. ACDC  
we use the same split and preprossing with [SSL4MIS]:https://github.com/HiLab-git/SSL4MIS. 
2. ISIC  
ISIC2018 can be downloaded here [link]:https://www.isic-archive.com/

## Enviorments
- python 3.7
- pytorch 1.8.1
- torchvision 0.9.1  

## Train  
For ACDC dataset:
'''
python train_MedTrans.py
'''
For ISIC dataset:
'''
python train_MedTrans_ISIC.py
'''
## Test
'''
python test_2D_fully.py
'''
## Reference
- [SSL4MIS](https://github.com/HiLab-git/SSL4MIS)
- [BAT](https://github.com/jcwang123/BA-Transformer)
