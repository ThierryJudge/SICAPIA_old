
---   
<div align="center">    
 
# SICAPIA   
<!--
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  

ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->



<!--  
Conference   
-->   
</div>
 
## Description   

Active learning and semi-supervised learning.

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/ThierryJudge/SICAPIA.git 

#create conda environement
conda env create -f sicapia.yml

# Activate conda environement
conda activate sicapia

# For CPU pytorch 
conda uninstall pytorch torchvision 
conda install pytorch torchvision cpuonly -c pytorch

# Setup projet
python setup.py develop
 ```
   
 Next, navigate to [Your Main Contribution (MNIST here)] and run it.   
 ```bash
# module folder
cd sicapia/   

# run module 
python ActiveLearningSystem.py
```

## Contributions Notes

### Docstring 
Use Google docstring format 
 ```
    def foo(self, arg1, arg2):
        """
            Function descriptioon 
        Args:
            arg1: type, description
            arg2: type, description

        Returns:
            type, description 
        """
```
In Pycharm, go to **Settings | Tools | Python Integrated Tools | Docstring format** and change to c

<!--
## Main Contribution      
List your modules here. Each module contains all code for a full system including how to run instructions.   
- [MNIST](https://github.com/williamFalcon/pytorch-lightning-conference-seed/tree/master/research_seed/mnist)  

## Baselines    
List your baselines here.   
- [MNIST_baseline](https://github.com/williamFalcon/pytorch-lightning-conference-seed/tree/master/research_seed/baselines/mnist_baseline)  

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```  
--> 
