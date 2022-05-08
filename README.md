# Lip-Sync-STS
When we translate a video from one language to another, lips get out of sync due to characteristics of different languages. So, in this project, we built a model to Sync the Lips in accordance with the Audio. The project was divided into two tasks-

Task 1- Translate the audio from English to Hindi.
Task 2- Sync the Lips of video according to generated Hindi audio.

## Task 1
In this task, for speech-to-speech translation, we use an automatic speech recognizer (ASR) to transcribe text from the original speech in Hindi language. We adapt neural machine translation and text-to-speech models to work for Indian languages and generate translated speech. The Task 1 proceeds in following way: Video  -> English Audio  -> English Text  -> Hindi Text  -> Hindi Audio.
The input given is the video with audio in English language. We extracted English audio from video using Moviepy library. Then converted this audio to text using Deepvoice 2 model of Pytorch. Then translated this English text to Hindi Text using a language translation model of huggingface. There was limit of 5000 bytes for this model, so we broke the text into several chunks of 2000 bytes and then applied the model. After that using Google API, we generated voice from the translated Hindi text and will merge it using Lip GAN. We adopt this approach to achieve high quality text-to-speech synthesis in our target language.
https://colab.research.google.com/drive/1scp7LqkbFx5QghIPU3W4F-7771BzNV9S?usp=sharing

![image](https://user-images.githubusercontent.com/79749572/167292696-cd46db8e-8000-4a7d-80b9-f6e8b4054382.png)

## Task 2
In this task we are given with a source or input video and a translated audio speech in Hindi. The translated audio is created using the English audio or speech from given input video file. Our task is to generate a lip-synced video having speech language as Hindi using these two inputs. 
This task consists of a GAN network having a generator to generate lip-synced frames and discriminator to check whether lip-synced occurred or not. Both are discussed in detail later on 

### Model Formulation
In a nutshell, our setup contains two networks, a generator G that generates faces by conditioning on audio inputs and a discriminator D that tests whether the generated face and the input audio are in sync. By training these networks together, the generator G learns to create photo-realistic faces that are accurately in sync with the given input audio.

![image](https://user-images.githubusercontent.com/79749572/167292799-228bb906-a34d-4414-9d78-1d440719ebc1.png)

#### Generator
The generator network contains three branches: 
(i) Face encoder  (ii) Audio encoder  (iii) Face Decoder.
The Face Encoder
During the training process of the generator , a face image of random pose and its corresponding audio segment is given as input and the generator is expected to morph the lip shape. Along with the random identity face image I, we also provide the desired pose information of the ground-truth as input to the face encoder. We mask the lower half of the ground truth face image and concatenate it channel-wise with I. 
The masked ground truth image provides the network with information about the target pose while ensuring that the network never gets any information about the ground truth lip shape. Thus, our final input to the face encoder , a regular CNN, is a HxHx6 image. 
Audio Encoder
The audio encoder is a standard CNN that takes a Mel-frequency cepstral coefficient (MFCC) heatmap of size MxTx1 and creates an audio embedding of size h. The audio embedding is concatenated with the face embedding to produce a joint audio-visual embedding of size 2xh.
Face Decoder
This branch produces a lip-synchronized face from the joint audio-visual embedding by inpainting the masked region of the input image with an appropriate mouth shape. It contains a series of residual blocks with a few intermediate deconvolutional layers. The output layer of the Face decoder is a sigmoid activated 1x1 convolutional layer with 3 filters, resulting in a face image of HxHx3.

#### Discriminator
We used L2 reconstruction loss for the generator that generated satisfactory talking faces, employing strong additional supervision can help the generator learn robust, accurate phoneme viseme mappings and make the facial movements more natural. We are directly testing whether the generated face synchronizes with the audio provides a stronger supervisory signal to the generator network. Accordingly, we create a network that encodes an input face and audio into fixed representations and computes the L2 distance d between them. The face encoder and audio encoder are the same as used in the generator network. The discriminator learns to detect synchronization by minimizing the following contrastive loss:
![image](https://user-images.githubusercontent.com/79749572/167292922-279dc3ad-14dc-4818-9afc-da123aa22832.png)


## Result
Input Video
![image](https://user-images.githubusercontent.com/79749572/167292948-358f249f-29a4-4a2e-b922-a9222574fcef.png)

Translated Audio
![image](https://user-images.githubusercontent.com/79749572/167292962-aa54744a-12f5-4bc6-abac-6ff305c98197.png)

Final Output
![image](https://user-images.githubusercontent.com/79749572/167292972-a71c3442-c1ad-4aec-aaec-30a992b2dec5.png)


## Applications
![image](https://user-images.githubusercontent.com/79749572/167292981-27bfc583-c9e5-44ef-9483-23250d6a861b.png)

#### Reference Papers
http://cvit.iiit.ac.in/research/projects/cvit-projects/facetoface-translation
https://arxiv.org/format/1611.01599
https://arxiv.org/pdf/2008.10010v1.pdf
















