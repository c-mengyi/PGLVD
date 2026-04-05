# PGLVD

Code relaese for [Prototype-Guided Local View Design for Open-Set Few-Shot Face Recognition]

## Setup

### Code environment

* You can create a conda environment with the correct dependencies using the following command lines:

  ```shell
      conda env create -f environment.yaml
      conda activate PGLVD
  ```

### Download Pretrained Weights

The VGGNet-19 and ResNet-50 models were selected as encoders.

- <a href="https://drive.google.com/file/d/1tGoX7fR-8m8MufA7HQdWWQn-DgxEOYJK/view?usp=share_link" target="_blank">VGGNet-19</a>
- <a href="https://drive.google.com/file/d/1aniiywJB-1jJRuq-vdpxAnKPp38y1CF3/view?usp=share_link" target="_blank">ResNet-50</a>

### Datasets

The pretrain set is VGGFace2, and the evaluation set can be either IJBC or CASIA-WebFace.

#### Download Dataset:

- <a href="https://pan.baidu.com/s/1c3KeLzy" target="_blank"> VGGFace2</a>

Above is the VGGFace2 dataset before processing, which needs to be processed to retain only identities with more than 150 images.

- <a href="https://drive.google.com/file/d/1BSdGyJn0mTWuZDA-_fo7eQvElH2qD2X9/view?usp=share_link" target="_blank"> CASIA-WebFace</a>

Above is the CASIA-WebFace dataset, with images already cropped using the <a href="https://github.com/timesler/facenet-pytorch" target="_blank">MTCNN by timesler</a>.

- <a href="https://drive.google.com/file/d/1aC4zf2Bn0xCVH_ZtEuQipR2JvRb1bf8o/view" target="_blank"> IJB-C</a>

Above is the IJB-C dataset before processing, which should be processed in ```IJBC_process.py``` to achieve the structure ```ROOT/SUBJECT_NAME/image.jpg```.

The face dataset must have the structure ```ROOT/SUBJECT_NAME/image.jpg```. To use your own dataset, please refer to [Your Dataset, Click Here!](./docs/prepare_custom_dataset.md).

After downloading, change the ```dataset_config``` and ```encoder_config``` in ```config.py``` accordingly.


## Usage
After the setup is done, simply run:
  ```shell
      python main_c2f.py --encoder="Res50" --dataset="IJBC" --T_L=0.4 --J=4 --c_k=25 --alpha=0.9 --tau=15 --th=0.6
      python main_c2f.py --encoder="Res50" --dataset="CASIA" --T_L=0.5 --J=4 --c_k=30 --alpha=0.9 --tau=15 --th=0.5
      python main_c2f.py --encoder="VGG19" --dataset="IJBC" --T_L=0.5 --J=4 --c_k=25 --alpha=0.9 --tau=15 --th=0.6
      python main_c2f.py --encoder="VGG19" --dataset="CASIA" --T_L=0.4 --J=4 --c_k=30 --alpha=0.85 --tau=20 --th=0.5
  ```

## References

Thanks to  [Hojin](https://github.com/1ho0jin1/OSFI-by-FineTuning) for the preliminary implementations.

## Contact

Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:

- mengyichen@email.ncu.edu.cn