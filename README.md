# S-DCGAN
A simple Tensorflow code for Deep convolutional GAN
## Requirements
Tensorflow 1.4(>1.0)

Python 2.7(>2.7)

Data: CUB200-2011(11788 birds) http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

## Details
This implementation is a little different from the paper <UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS>. 

Discriminator network:

   Input: [BATCH_SIZE, 64, 64 ,3]
   
   conv1: [BATCH_SIZE, 32, 32, 64]
   
   conv2: [BATCH_SIZE, 16, 16, 128]
   
   conv3: [BATCH_SIZE, 8, 8, 256]
   
   conv4: [BATCH_SIZE, 4, 4, 512]
   
   reshape: [BATCH_SIZE, 8192]
   
   fc5: [BATCH_SIZE, 1]   

Generator network:

   Input: [BATCH_SIZE, 100]
   
   fc1: [BATCH_SIZE, 8192]
   
   reshape: [BATCH_SIZE, 4, 4, 512]
   
   conv_tran1: [BATCH_SIZE, 8, 8, 256]
   
   conv_tran2: [BATCH_SIZE, 16, 16, 128]
   
   conv_tran3: [BATCH_SIZE, 32, 32, 64]
   
   conv_tran4: [BATCH_SIZE, 64, 64, 3]

For ease of understanding, I fix all the values(i.e. input shape).
## Run
You need download data set CUB200-2011(http://www.vision.caltech.edu/visipedia/CUB-200-2011.html), and place images in 'CUB200-2011/images/'.

python train.py to run.
