# SENUCLS
This is the official PyTorch implementation of paper - <Structure Embedded Nucleus Classification for Histopathology Images>, a graph neural network based method for nuclei classification.

> **If you intend to use anything from this repo, citation of the original publication given above is necessary**

## Set Up Environment
```
conda env create -f environment.yml
conda activate hovernet
pip install torch==1.10.0 torchvision==0.11.1
pip install torch-geometric torch-scatter torch-sparse
```

## Datasets
<!-- We released a new nuclei segmentation and classification dataset called FFPE-CRC dataset. The images are 1000 x 1000 patches from the TCGA dataset (The Cancer Genome Atlas. https://tcga-data.nci.nih.gov/docs/publications/tcga) and labeled by experts in our local hospitals. The nuclei types of the CRC-FFPE dataset include Tumor, Stroma, Immune, Necrosis, and Other. These images are divided into a training set (45 tiles) and a testing set (14 tiles). -->
- [CoNSeP](https://www.sciencedirect.com/science/article/pii/S1361841519301045)
- [PanNuke](https://arxiv.org/abs/2003.10778)
- [MoNuSAC](https://ieeexplore.ieee.org/abstract/document/8880654)
- [FFPE]()

# Running the Code

## Training

### Data Format
For training, patches must be extracted using `extract_patches.py`. For each patch, patches are stored as a 4 dimensional numpy array with channels [RGB, inst]. Here, inst is the instance segmentation ground truth. I.e pixels range from 0 to N, where 0 is background and N is the number of nuclear instances for that particular image. 

Before training:

- Set path to the data directories in `config.py`
- Set path where checkpoints will be saved  in `config.py`
- Set path to pretrained VAN-base weights in `models/hovernet/opt.py`. Download the weights [here](https://drive.google.com/file/d/1ne9rpzimYh7EyaUU5kfDd2nDzl04LJ5v/view?usp=sharing).
- Modify hyperparameters, including number of epochs and learning rate in `models/hovernet/opt.py`.

- To initialise the training script with GPUs 0, the command is:
```
python run_train.py --gpu='0,1' 
```

## Inference

### Data Format

Input: <br />
- Standard images files, including `png`, `jpg` and `tiff`.
- WSIs supported by [OpenSlide](https://openslide.org/), including `svs`, `tif`, `ndpi` and `mrxs`.
- Instance segmentation results output from other methods, like HoverNet or MaskRCNN. The formats of the segmentation results are '.mat'. The filename should match the testing images.

### Inference codes for tiles
```
python -u run_infer.py \
--gpu='0' \
--nr_types=6 \ # number of types + 1
--type_info_path=type_info.json \
--batch_size=1 \
--model_mode=original \
--model_path=.tar \ # choose the trained weights
--nr_inference_workers=1 \
--nr_post_proc_workers=16 \
tile \
--input_dir='PaNuKe/Fold3/images/' \ # testing tile path
--output_dir=panuke_out/ \  # output path
--inst_dir='inst_prediction/' \ # instance segmentation results path
--mem_usage=0.1 \
--save_qupath
```
Output: : <br />
- mat files / JSON files : Including centroid coordinates and nuclei types.
- overlay images: Visualization of the classification results.

### Inference codes for WSI
```
python run_infer.py \
--gpu='0' \
--nr_types=6 \ # number of types + 1
--type_info_path=type_info.json \
--batch_size=1 \
--model_mode=original \
--model_path=.tar \ # choose the trained weights
--nr_inference_workers=1 \
--nr_post_proc_workers=0 \
wsi \
--input_dir='test/wsi/' \ # testing wsi path
--output_dir='wsi_out/' \ # output path
--inst_pred_dir='test/inst_pred/' \ # instance segmentation results path
--proc_mag=20 \
--input_mask_dir='test/msk/' \
--save_thumb \
--save_mask
```
Output: : <br />
- JSON files : Including centroid coordinates and nuclei types.

Post process to .svs file: <br />
```
python prediction2svs.py # change input file name in the codes
```


## Citation

If any part of this code is used, please give appropriate citation to our paper. <br />

BibTex entry: <br />
```
@misc{lou2023structure,
      title={Structure Embedded Nucleus Classification for Histopathology Images}, 
      author={Wei Lou and Xiang Wan and Guanbin Li and Xiaoying Lou and Chenghang Li and Feng Gao and Haofeng Li},
      year={2023},
      eprint={2302.11416},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
