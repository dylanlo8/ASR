from Whisper import Whisper
from Processor import Processor
from TranslateModel import TranslateModel
from Orchestrator import AudioEmbeddingsDataset
import torch
from torch.utils.data import DataLoader

def main():
    list_of_audio_files = ["data/sub/De95Osq7p1c_trimmed_segment_1.wav"]
    labels = [""]
    
    # Process audio
    processor = Processor()
    train_dataset = processor.process_audio(list_of_audio_files, labels)
    
    # Embed audio
    whisper = Whisper()
    train_audio_embeddings, train_labels = whisper.embed_audio(train_dataset)
    
    torch.cuda.empty_cache()
    
    # Translate embeddings
    translate_model_inst = TranslateModel()
    output, train_labels = translate_model_inst.forward(train_audio_embeddings)
    results = translate_model_inst.decode(output)

    # Dataloader sample code for fun
    train_audiodataset = AudioEmbeddingsDataset(train_audio_embeddings, train_labels)
    train_audioloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    print(results)
    return results

if __name__ == "__main__":
    main()
