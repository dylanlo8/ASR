from Whisper import Whisper
from Processor import Processor
from TranslateModel import TranslateModel
import torch

def main():
    list_of_audio_files = ["data/sub/De95Osq7p1c_trimmed_segment_1.wav"]
    
    # Process audio
    processor = Processor()
    audio_dataset, att = processor.process_audio(list_audio_filepaths=list_of_audio_files)
    
    # Embed audio
    whisper = Whisper()
    embeddings = whisper.embed_audio(audio_dataset.to("cuda"), att)
    
    # Translate embeddings
    translate_model_inst = TranslateModel()  # Change instance name
    output = translate_model_inst.forward(embeddings)
    results = translate_model_inst.decode(output)

    print(results)
    return results

if __name__ == "__main__":
    main()
