# FireRedASR

[[Blog]](https://fireredteam.github.io/demos/firered_asr/)
[[Paper]]()
[[Model]](https://huggingface.co/fireredteam)

FireRedASR, a family of large-scale automatic speech recognition (ASR) models for Mandarin, designed to meet diverse requirements in superior performance and optimal efficiency across various applications. FireRedASR comprises two variants:
- FireRedASR-LLM: Designed to achieve state-of-the-art (SOTA) performance and to enable seamless end-to-end speech interaction. It adopts an Encoder-Adapter-LLM framework leveraging large language model (LLM) capabilities.
- FireRedASR-AED: Designed to balance high performance and computational efficiency and to serve as an effective speech representation module in LLM-based speech models. It utilizes an Attention-based Encoder-Decoder (AED) architecture.

![Model](/assets/FireRedASR_model.png)


## News
- [2025/01/24] 🔥 We release [techincal report]()(under review at arXiv), [blog](https://fireredteam.github.io/demos/firered_asr/), and [FireRedASR-AED-L](https://huggingface.co/fireredteam/FireRedASR-AED-L/tree/main) model weights.
- [WIP] We plan to release FireRedASR-LLM-L after the Spring Festival.

## Setup

```bash
$ git clone https://github.com/FireRedTeam/FireRedASR.git
$ conda create --name fireredasr python=3.10
$ pip install -r requirements.txt
```

## Usage
Download model files from [huggingface](https://huggingface.co/fireredteam) and place them in the folder `pretrained_models`

### Quick Start
```bash
$ cd examples/
$ bash inference_fireredasr_aed.sh
$ bash inference_fireredasr_llm.sh
```

### Commond-line Usage
```bash
# Setup PATH & PYTHONPATH
$ export PATH=$PWD/fireredasr/:$PWD/fireredasr/utils/:$PATH
$ export PYTHONPATH=$PWD/:$PYTHONPATH
$ speech2text.py --help
$ speech2text.py --wav_path examples/wav/BAC009S0764W0121.wav --asr_type "aed" --model_dir pretrained_models/FireRedASR-AED-L
$ speech2text.py --wav_path examples/wav/BAC009S0764W0121.wav --asr_type "llm" --model_dir pretrained_models/FireRedASR-LLM-L
```

### Python Usage
```python
from fireredasr.models.fireredasr import FireRedAsr

batch_uttid = ["BAC009S0764W0121"]
batch_wav_path = ["examples/wav/BAC009S0764W0121.wav"]

# FireRedASR-AED
model = FireRedAsr.from_pretrained("aed", "pretrained_models/FireRedASR-AED-L")
results = model.transcribe(
    batch_uttid,
    batch_wav_path,
    {
    "use_gpu": 1,
    "beam_size": 3,
    "nbest": 1,
    "decode_max_len": 0,
    "softmax_smoothing": 1.0,
    "aed_length_penalty": 0.0,
    "eos_penalty": 1.0
    }
)
print(results)


# FireRedASR-LLM
model = FireRedAsr.from_pretrained("llm", "pretrained_models/FireRedASR-LLM-L")
results = model.transcribe(
    batch_uttid,
    batch_wav_path,
    {
    "use_gpu": 1,
    "beam_size": 3,
    "decode_max_len": 0,
    "decode_min_len": 0,
    "repetition_penalty": 1.0,
    "llm_length_penalty": 0.0,
    "temperature": 1.0
    }
)
print(results)
```


## Acknowledgements
Thanks to the following open-source works:
- [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)
- [icefall/ASR_LLM](https://github.com/k2-fsa/icefall/tree/master/egs/speech_llm/ASR_LLM)
- [WeNet](https://github.com/wenet-e2e/wenet)
- [Speech-Transformer](https://github.com/kaituoxu/Speech-Transformer)
