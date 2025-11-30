# Music Style Transfer
This project aims to explore transferring stylistic traits, such as texture, timbre, rhythm, and harmony from a piece of music 
in one genre to another piece from a different genre using both supervised and unsupervised learning.

## Installation
To install, run the command to create a conda environment:
```python
conda env create  -f environment.yml
```

## Preprocessing
This project uses the FMA small dataset from the [FMA repository](https://github.com/mdeff/fma). To run this program, you must install both the [FMA small dataset](https://os.unil.cloud.switch.ch/fma/fma_small.zip), 
along with the [FMA metadata](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip) from the repository, and place those in a data folder which can be specified in the configs.

The FMA dataset and metadata should be placed in a /data folder, unless specified otherwise with the config arguments. The default folder structure is:
```
├───data
    ├───fma_metadata
    ├───fma_small
```
Unless otherwise specified with the arguments when running the models:
```python
--dataset_dir <directory>
--metadata_dir <directory>
```
To preprocess the data, we must create a Mel spectrogram numpy file of each audio file, which can be done with the helper file compute_mels. 
It only needs to be run once before running the main file for either model. It can be run with:
```python
python compute_mels.py
```
Which will create a /mels directory in the `dataset_dir` folder, along with creating a mels.csv inside the metadata folder

## Supervised Learning
Supervised learning is performed via a CNN using paired images to create one generated audio file and spectrogram. 
It takes in 3 separate spectrograms, a content, style, and input spectrogram. 

The content is the original genre to be transferred from (`--genre_A` in config), style is the genre to be transferred (`--genre_B` in config), 
and the input is the input where the style will be transferred to, and used for generation.

Random files will be chosen for this learning by using the seed, which can be changed with the `--seed` config, for reproducibility.

## Unsupervised Learning
Unsupervised learning is done through a Generative Adversarial Network architecture known as CycleGAN. This features two Generators, 
one of which will transfer from genre A to B, and the other from B to A, alongside two discriminators, 
which are meant to distinguish between real and generated data.
Both the generator and discriminator are trained together, while the generator learns to create more realistic samples as it
tries to get past the discriminator, which also gets better at distinguishing the inputs.

This architecture works off of unpaired data, making it more complex since we have must train the generators on the two genres before we can create outputs,
compared to immediately creating generated audio with our supervised learning approach.

The generators are two U-Nets, while the discriminators are patchGANs, which focus on high frequency details to determine whether the given input is real or fake.

## Training and Running Models
After creating the spectrogram files, you can run either the CNN or CycleGAN models with:
```python
python main_cnn.py
python main_cyclegan.py
```
This will create generated mel spectrogram files as .npy files, alongside generated audio files in the output folder 
/outputs unless specified with the command `--ouput_dir <directory>`. 
To turn the spectrograms into images for visual analysis, you may run the helper script:
```python
python mel_images.py
```
Which will create png plots using matplotlib for:
- The content audio, style audio, and generated audio for the CNN model
- The 10 generated audios, 5 from genre A to B, 5 from B to A, along with the difference between the original and the generated file for those same generations.

The images are in the decibel scale, and it will also print out the numerical difference between the original and generated in decibels.