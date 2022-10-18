# ASR project

Deep Learning for Audio, HSE

## About

This repo contains DeepSpeech2-like model implementation ([link](https://arxiv.org/abs/1512.02595)), trained on 960h of transcribed audio from [Librispeech](https://www.openslr.org/12).

## Installation guide

First, clone the repo and set up the environment

```shell
git clone https://github.com/Mikezz1/asr.git
sh asr/setup.sh
sh asr/setup_lm.sh
```

To run on yandex datasphere you may also  need to initialise wandb first:

```python
import wandb
wandb.init()
```

## Usage

To run training use

```shell
!python3 asr/train.py --resume='asr/final_model/model_best-6.pth' \
                --config='asr/hw_asr/configs/ds2-librispeech-batch-overfit.json'
```

To run final evaluation use the following command:

```shell
!python3 asr/test.py --resume='asr/final_model/model_best-6.pth' \
                --config='asr/hw_asr/configs/ds2-librispeech-test.json' \
                --batch-size=1 \
                --evaluate=True \
                --beam_size=50 \
                --beam_search_type='custom' \
                --output="test_result.json"
```

`evaluate=True` argument enables WER/CER calculation, `beam_search_type` takes either "custom", "torch", or "fast" values. Final metric was calculated with "custom" beam search. `beam_size`  controls num. beams in beam search.  If you want to use LM with custom beamseach, make sure that you've specified decoder.path_to_lm in test json config.

Pls find training logs [here](https://wandb.ai/mikezz1/asr_project?workspace=user-mikezz1) and model weights [here](https://drive.google.com/file/d/1fA3MNHDkO-ThK2tEIng_o6VXstXtF93I/view?usp=sharing)
