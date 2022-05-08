# Lip-Sync-STS
When we translate a video from one language to another, lips get out of sync due to characteristics of different languages. So, in this project, we built a model to Sync the Lips in accordance with the Audio. To keep the project uncomplicated project was divided into two tasks-

Task 1- Translate the audio from English to Hindi.<br/>
Task 2- Sync the Lips of video according to generated Hindi audio.

## Task 1
In this task, for speech-to-speech translation, we use an automatic speech recognizer (ASR) to transcribe text from the original speech in Hindi language. We adapt neural machine translation and text-to-speech models to work for Indian languages and generate translated speech. The Task 1 proceeds in following way:<br/> Video  -> English Audio  -> English Text  -> Hindi Text  -> Hindi Audio.<br/>
The input given is the video with audio in English language. We extracted English audio from video using Moviepy library. Then converted this audio to text using Deepvoice 2 model of Pytorch. Then translated this English text to Hindi Text using a language translation model of huggingface. There was limit of 5000 bytes for this model, so we broke the text into several chunks of 2000 bytes and then applied the model. After that using Google API, we generated voice from the translated Hindi text and will merge it using Lip GAN. We adopt this approach to achieve high quality text-to-speech synthesis in our target language.<br/>
https://colab.research.google.com/drive/1scp7LqkbFx5QghIPU3W4F-7771BzNV9S?usp=sharing

<a href="url"><img src="https://user-images.githubusercontent.com/79749572/167292696-cd46db8e-8000-4a7d-80b9-f6e8b4054382.png" width="720" height = "350">

## Task 2
In this task we are given with a source or input video and a translated audio speech in Hindi. The translated audio is created using the English audio or speech from given input video file. Our task is to generate a lip-synced video having speech language as Hindi using these two inputs. <br/>
This task consists of a GAN network having a generator to generate lip-synced frames and discriminator to check whether lip-synced occurred or not. Both are discussed in detail later on 

## Model Formulation
In a nutshell, our setup contains two networks, a generator G that generates faces by conditioning on audio inputs and a discriminator D that tests whether the generated face and the input audio are in sync. By training these networks together, the generator G learns to create photo-realistic faces that are accurately in sync with the given input audio.<br/>
  
<a href="url"><img src="(https://user-images.githubusercontent.com/79749572/167292799-228bb906-a34d-4414-9d78-1d440719ebc1.png" width="720">

### Generator
The generator network contains three branches: <br/><br/>
(i) The Face Encoder<br/>
During the training process of the generator , a face image of random pose and its corresponding audio segment is given as input and the generator is expected to morph the lip shape. Along with the random identity face image I, we also provide the desired pose information of the ground-truth as input to the face encoder. We mask the lower half of the ground truth face image and concatenate it channel-wise with I. <br/><br/>
(i) The Face Encoder<br/>
The masked ground truth image provides the network with information about the target pose while ensuring that the network never gets any information about the ground truth lip shape. Thus, our final input to the face encoder , a regular CNN, is a HxHx6 image. <br/>
(ii) Audio Encoder<br/>
The audio encoder is a standard CNN that takes a Mel-frequency cepstral coefficient (MFCC) heatmap of size MxTx1 and creates an audio embedding of size h. The audio embedding is concatenated with the face embedding to produce a joint audio-visual embedding of size 2xh.<br/><br/>
(i) The Face Encoder<br/>
(iii) Face Decoder<br/>
This branch produces a lip-synchronized face from the joint audio-visual embedding by inpainting the masked region of the input image with an appropriate mouth shape. It contains a series of residual blocks with a few intermediate deconvolutional layers. The output layer of the Face decoder is a sigmoid activated 1x1 convolutional layer with 3 filters, resulting in a face image of HxHx3.<br/>

### Discriminator
We used L2 reconstruction loss for the generator that generated satisfactory talking faces, employing strong additional supervision can help the generator learn robust, accurate phoneme viseme mappings and make the facial movements more natural. We are directly testing whether the generated face synchronizes with the audio provides a stronger supervisory signal to the generator network. Accordingly, we create a network that encodes an input face and audio into fixed representations and computes the L2 distance d between them. The face encoder and audio encoder are the same as used in the generator network. The discriminator learns to detect synchronization by minimizing the following contrastive loss:<br/>
  <a href="url"><img src="(https://user-images.githubusercontent.com/79749572/167292922-279dc3ad-14dc-4818-9afc-da123aa22832.png" width="350">


## Result
#### Input Video

    
https://user-images.githubusercontent.com/79749572/167294455-815a2d20-f1e2-4cc3-a2ca-cd04bf169d9a.mp4

    
#### Translated Audio
    
    
<a href="url"><img src="(https://user-images.githubusercontent.com/79749572/167293718-952786b0-3a40-4c4e-b7da-2598a6bb7cf7.png" width="350">    

  
#### Final Output

  
https://user-images.githubusercontent.com/79749572/167294332-2a8aa36d-807e-475c-8ea0-88d14a3264b6.mp4

  
## Applications
![image](https://user-images.githubusercontent.com/79749572/167292981-27bfc583-c9e5-44ef-9483-23250d6a861b.png | width=200)

### Reference Papers
http://cvit.iiit.ac.in/research/projects/cvit-projects/facetoface-translation<br/>
https://arxiv.org/format/1611.01599<br/>
https://arxiv.org/pdf/2008.10010v1.pdf
















