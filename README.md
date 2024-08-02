# Automatic Speech Translation with PAD-Thai

This project aims to address the challenges of transcribing and translating audio from low-resource languages by leveraging the capabilities of large language models (LLMs) in a multimodal setting. By integrating the ASR and LLM with an projection adaptor, we sought to develop a novel solution for translating conversational low-resource languages. Our experiments have demonstrated the viability of using LLMs for direct audio-to-text translation in low-resource languages settings. We hope our work serves as a starting point for further exploration and improvement in this field, ultimately contributing to more accessible and accurate translation technologies for low-resource languages.

Our model architecture:
![ASR Diagram(2)](https://github.com/user-attachments/assets/2a38b960-1339-4fab-90fd-01c27f766bb5)

Our Adaptor architecture:
![ASR Diagram (3)](https://github.com/user-attachments/assets/16cc45b1-573b-4bc3-acba-d99d20ba98cf)

This repository contains the code necessary for training of an adaptor layer between a pre-trained ASR model (Whisper-Medium) and a pre-trained LLM (Meta-Llama-3.1-8B-Instruct) to complete the translation task. 

## Installations Required
Make sure that the models are installed locally on the machine before use. For our training, we have extracted our models from HuggingFace. 

For Whisper-Medium:
```
# Make sure that you have git-lfs installed
git lfs install
git clone https://huggingface.co/openai/whisper-medium
```

For Meta-Llama-3.1-8B-Instruct:
```
# Make sure that you have git-lfs installed
git lfs install
git clone https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
```

You can refer to the model's HuggingFace page for more details. 


## Training the Model
To train the model, modify `main.py` to set up the data to be used for training. The data should be a csv file consisting of the following two fields: eng_reference, trimmed_segment_path.

An example for reference: ```I may not have thought this way``` , ```data/sub/8p9rJ-cFHQ0_trimmed_segment_1.wav```

You can modify the other parameters in the file to suit your use case.


## To run Inference
You can run inference through the `predict.py` file. Specify the links to the audio files that you want to run inference on, and the relative path to the checkpoint of the model you are testing.
