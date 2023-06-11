# IsoRecon

Isotropic reconstruction of 3D EM images with unsupervised degradation learning



## Dependencies

python 3.6.x

torch>=0.4.1

If you have a [Docker](https://www.docker.com/) environment, we strongly recommend you to pull our image as follows,

```
docker pull shiyuuuu/cuda9.0-cudnn7-devel-ubuntu16.04-torch0.4.0:v1
```

or

```
docker pull registry.cn-hangzhou.aliyuncs.com/shiyu_666/cuda9.0-cudnn7-devel-ubuntu16.04-torch0.4:v1
```



## Training stage

```python
python main.py --dataroot PATH_TO_DATA --save PATH_TO_SAVE --cuda
```



## Testing stage

```python
python test.py --dataroot PATH_TO_TEST_DATA --modelpath PATH_TO_MODEL --cuda
```



## Acknowledgments

This project is built upon [CycleGAN](https://arxiv.org/abs/1703.10593) and its [cleaner version](https://github.com/aitorzip/PyTorch-CycleGAN).



## Contact

If you have any problem with the released code, please do not hesitate to contact me by email ([sydeng@mail.ustc.edu.cn](mailto:sydeng@mail.ustc.edu.cn)).
