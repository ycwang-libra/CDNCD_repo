# CDNCD: Cross Domain Novel Class Discovery
This is an implementation of our paper ["Exclusive Style Removal for Cross Domain Novel Class Discovery"](https://arxiv.org/abs/2406.18140)

## Usage
### Download the repository and install the requirements
- Clone this repo:
```bash
git clone git@github.com:ycwang-libra/CDNCD_repo.git
cd CDNCD_repo
```
- Create a Conda virtual environment and activate it:
```bash
conda create -n CDNCD python=3.8 -y
conda activate CDNCD
```
- Install frameworks: `PyTorch==1.13` and `torchvision==0.14` with `CUDA==11.6`
- Install toolboxes: `numpy==1.24.4`, `matplotlab==3.7.5`, `scikit-learn==1.3.2`
- Install other packages such as: `torchnet==0.0.4`, `wandb==0.17.7`, `tqdm==4.66.5`, etc

### Dataset preparation
If you want to train all algorithms in our papers. Please first download the original datasets [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html), [OfficeHome](https://www.hemanthdv.org/officeHomeDataset.html) and [DomainNet40](https://ai.bu.edu/M3SDA/), and then

* put the [datasets/OfficeHome/image_list](datasets/OfficeHome/image_list) and [datasets/DomainNet40/image_list](datasets/DomainNet40/image_list) into your downloaded OfficeHome and DomainNet40

* follow the [data_preparation/cifar10_corrupt.py](data_preparation/cifar10_corrupt.py) and [data_preparation/officehome_real_corrupt.py](data_preparation/officehome_real_corrupt.py) to construct the corrupt CIFAR10 and OfficeHome data (Real_World domain)

### Training and Testing
Follow commands below to train main model (improved SimGCD) with different datasets
```bash
cd CDNCD
sh scripts/train_test_cifar10cmix_5-5_corrupt.sh # on cifar10cmix data
sh scripts/train_test_cifar10call_5-5_corrupt.sh # on cifar10call data
sh scripts/train_test_officehomecmix_20-20.sh # on OfficeHomecmix data
sh scripts/train_test_officehome_20-20_cross_domain.sh # on OfficeHome cross domain setting
sh scripts/train_test_domainnet40_20-20_cross_domain.sh # on DomainNet40 cross domain setting
```
We also provide other NCD algorithms RS, UNO and ComEx with/without style remove module in [SOTAs](SOTAs), so you can also train these models with the same backbone (same backbone only on UNO and ComEx).

## Citing CDNCD
If you find this work useful for your research, please consider citing our [paper](https://arxiv.org/abs/2406.18140):

```bibtex
@article{wang2024exclusive,
  title={Exclusive Style Removal for Cross Domain Novel Class Discovery},
  author={Wang, Yicheng and Liu, Feng and Liu, Junmin and Sun, Kai},
  journal={arXiv preprint arXiv:2406.18140},
  year={2024}
}
```