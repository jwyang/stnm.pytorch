# stnm.pytorch
Pytorch code for spatial transformer network with mask (STNM). Given the canvas, foreground image, foreground mask and the transformation for the foreground, STNM paste the foreground after transformed to the canvas with the alpha blending guided by the foreground mask. This is used in our paper "LR-GAN: Layered Recursive Generative Adversarial Networks for Image Generation".

### Build

1. Install [PyTorch](http://pytorch.org/) with proper commands.

2. Go to the folder script, and them simply run the following commands:
```bash
$ ./make.sh
$ python test.sh
```
If there is no errors by far, then congratulations, you make it!

### Citation
If you find our code is useful in your researches, please cite:

@article{yang2017lr,
  title={LR-GAN: Layered recursive generative adversarial networks for image generation},
  author={Yang, Jianwei and Kannan, Anitha and Batra, Dhruv and Parikh, Devi},
  journal={ICLR},
  year={2017}
}

@inproceedings{jaderberg2015spatial,
  title={Spatial transformer networks},
  author={Jaderberg, Max and Simonyan, Karen and Zisserman, Andrew and others},
  booktitle={Advances in Neural Information Processing Systems},
  pages={2017--2025},
  year={2015}
}

### Reference

This project is built based on [pytorch-stn](https://github.com/fxia22/stn.pytorch).
