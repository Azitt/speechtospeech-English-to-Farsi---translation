#import numpy as np
import pandas as pd
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
import torchaudio
import matplotlib
import time
from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
from scipy.io.wavfile import write
from flask import Flask, send_file
import os.path



class _STS_model:
    
    model0 = None
    processor0 = None  
    model1 = None
    tokenizer1 = None 
    tag = None
    vocoder_tag = None
    text2speech = None 
    _instance = None


    def predict(self, file_path):
        """
        :param file_path (str): Path to inpute file to predict
        :return otput speech in English by the model
        """
        ####SpeechtoText part####################################################

        resampler = torchaudio.transforms.Resample(orig_freq=48_000, new_freq=16_000)
        batch = {}
        speech, _ = torchaudio.load(file_path)
        batch["speech"] = resampler.forward(speech.squeeze(0)).numpy()
        batch["sampling_rate"] = resampler.new_freq

        data= batch #self.load_file_to_data("test/001001.wav")  # '/home/azam/capstone/001001.wav'
        print("Before Speech to text...")
        input_dict = self.processor0(data["speech"], sampling_rate=data["sampling_rate"], return_tensors="pt", padding=True)
        logits = self.model0(input_dict.input_values.to("cpu")).logits
        pred_ids = torch.argmax(logits, dim=-1)[0]
        st_out = self.processor0.decode(pred_ids)
        print(st_out)
        print("After Speech to text...")
        ####TexttoText part######################################################
        self.tokenizer1.src_lang = "fa_IR"
        encoded_hi = self.tokenizer1(st_out, return_tensors="pt")
        generated_tokens = self.model1.generate(
            **encoded_hi,
            forced_bos_token_id = self.tokenizer1.lang_code_to_id["en_XX"]
        )
        out_translate = self.tokenizer1.batch_decode(generated_tokens, skip_special_tokens=True)
        out_translates = ''.join(str(e) for e in out_translate)
        print(out_translates)
        print("After text to text...")
        ###TexttoSpeech #############################################################

        x = out_translates #"It's dedicated to God, the creator of the universe"#input()
        # synthesis
        with torch.no_grad():
              start = time.time()
              wav2 = self.text2speech(x)["wav"]
              rtf = (time.time() - start) / (len(wav2) / self.text2speech.fs)
        print(f"RTF = {rtf:5f}")
        #speech = self.text2speech("foobar")["wav"]
        out_path = f'static/audio/outputs/{os.path.basename(file_path)}'
        write(out_path, self.text2speech.fs, wav2.view(-1).cpu().numpy())
        # write(out_path, self.text2speech.fs, speech.numpy())
        print("After text to speech...")
        return [st_out,out_translates,out_path]
       # text2speech = Text2Speech.from_pretrained("model_name")        
        #  soundfile.write("out.wav", speech.numpy(), text2speech.fs, "PCM_16")
def STS_model():
    if _STS_model._instance is None:
       _STS_model._instance = _STS_model()
       _STS_model.model0 = Wav2Vec2ForCTC.from_pretrained("/app/checkpoint-4000")  #../checkpoint-4000
       _STS_model.processor0 = Wav2Vec2Processor.from_pretrained("/app/checkpoint-4000")
       _STS_model.model1 = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
       _STS_model.tokenizer1 = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
       _STS_model.text2speech = Text2Speech.from_pretrained(
            model_tag=str_or_none('kan-bayashi/ljspeech_fastspeech2'),
            vocoder_tag=str_or_none("parallel_wavegan/ljspeech_hifigan.v1"),
            device="cpu",
            threshold=0.5,
            minlenratio=0.0,
            maxlenratio=10.0,
            use_att_constraint=False,
            backward_window=1,
            forward_window=3,
            speed_control_alpha=1.0,
            noise_scale=0.333,
            noise_scale_dur=0.333,
        )
    return _STS_model._instance 




if __name__ == "__main__":

    # create 2 instances of the model
    kss = STS_model()
    kss1 = STS_model()

    assert kss is kss1
    
  