# feifeifanclub - Neural Style Transfer with Learned Artist Representation!

Learning the style of a whole artist instead of simply the style of a specified image :)

## How to set up:
  - Download /models/, /images/, and /dicts/ folders from ________, and place them in the repository folder.
  - Then run inference.py using images from /cuteimages/ or your own images by setting CONTENT_IMG to the filename and setting other parameters! 
  - To train a transfer model using one of the following approaches, use train_style_transfer.py
     - 'average' (raw RGB pixel average across artist)    
     - 'smartaverage' (average style features across artist, extracted from VGG)     
     - 'random' (random painting from artist)             
     - 'cycle' (cycle artist paintings during training)        
     - 'classifier' (backprop through pre-trained artist classfier)
     
     
 
## Acknowledgements:
  - Painting dataset gotten from the following Kaggle competiton:
        https://www.kaggle.com/ikarus777/best-artworks-of-all-time
  - Dataset of random images to use as arbitrary content images gotten from:
        https://github.com/fastai/imagenette
  - The baseline architecture is gotten from:
        J. Johnson, A. Alahi, and L. Fei-Fei, “Perceptual losses for real-time style transfer and super-resolution,”
        inComputer Vision – ECCV 2016, B. Leibe, J. Matas, N. Sebe, and M. Welling, Eds.Cham: Springer InternationalPublishing, 2016, pp. 694–711
  - Lots of code for the implementation of the baseline was adapted from:
        R. Mina, “fast-neural-style: Fast style transfer in pytorch!” https://github.com/iamRusty/fast-neural-style-pytorch,2018
  - Pre-trained VGG feature extractor state_dict gotten from:
        https://web.eecs.umich.edu/~justincj/models/vgg16-00b39a1b.pth
  - Pre-trained ResNet-50 artist classfier state_dict gotten from the following Kaggle submission:
        https://www.kaggle.com/attol8/paintings-classifier-fastai-resnet50-90-2-ac%20c/notebook
  - Love for computer vision gotten from:
        Prof. Olga Russakovsky! :)    (ooh and Prof. Fei Fei Li)
