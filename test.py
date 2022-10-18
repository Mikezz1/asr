import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

import hw_asr.model as module_model
from hw_asr.trainer import Trainer
from hw_asr.utils import ROOT_PATH
from hw_asr.utils.object_loading import get_dataloaders
from hw_asr.utils.parse_config import ConfigParser
# from torchaudio.models.decoder import ctc_decoder
from string import ascii_lowercase
from hw_asr.metric.utils import calc_cer, calc_wer
import torchaudio
from torchaudio.models.decoder import ctc_decoder
from torchaudio.models.decoder import download_pretrained_files

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file, eval=False, beam_size=100, beam_search_type='torch'):
    logger = config.get_logger("test")

    files = None
    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # text_encoder
    text_encoder = config.get_text_encoder()

    # setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)

    # build model architecture
    model = config.init_obj(
        config["arch"],
        module_model, n_class=len(text_encoder))
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    results = []
    cers_argmax, wers_argmax, cers_bs, wers_bs = [], [], [], []

    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(dataloaders["test"])):
            batch = Trainer.move_batch_to_device(batch, device)
            output = model(**batch)
            if type(output) is dict:
                batch.update(output)
            else:
                batch["logits"] = output
            batch["log_probs"] = torch.log_softmax(batch["logits"], dim=-1)
            batch["log_probs_length"] = model.transform_input_lengths(
                batch["spectrogram_length"]
            )
            batch["probs"] = batch["log_probs"].exp().detach().cpu()
            batch["argmax"] = batch["probs"].argmax(-1)

            if beam_search_type == 'torch':
                files = download_pretrained_files("librispeech-4-gram")
                encoder = text_encoder.ctc_beam_search_pt
            elif beam_search_type == 'fast':
                encoder = text_encoder.ctc_beam_search_fast
            elif beam_search_type == 'custom':
                encoder = text_encoder.ctc_beam_search
            else:
                raise NotImplementedError

            for i in range(len(batch["text"])):
                argmax = batch["argmax"][i]
                argmax = argmax[: int(batch["log_probs_length"][i])]

                results.append(
                    {"ground_trurh": batch["text"][i],
                     "pred_text_argmax": text_encoder.ctc_decode(
                         argmax.cpu().numpy()),
                     "pred_text_beam_search": encoder(
                         batch["probs"][i],
                         batch["log_probs_length"][i],
                         beam_size=beam_size, files=files)[: 10], })
            if eval:
                for sample in results:
                    cers_argmax.append(
                        calc_cer(
                            sample['ground_trurh'],
                            sample['pred_text_argmax']))
                    cers_bs.append(
                        calc_cer(
                            sample['ground_trurh'],
                            sample['pred_text_beam_search'][0].text))
                    wers_argmax.append(
                        calc_wer(
                            sample['ground_trurh'],
                            sample['pred_text_argmax']))
                    wers_bs.append(
                        calc_wer(
                            sample['ground_trurh'],
                            sample['pred_text_beam_search'][0].text))

        if eval:
            print('-' * 70)
            print(
                f"CER-argmax: {np.mean(cers_argmax)},  CER-beamsearch: {np.mean(cers_bs)}"
                f"\nWER-argmax: {np.mean(wers_argmax)}, WER-beamsearch: {np.mean(wers_bs)}")
            print('-' * 70)

    with Path(out_file).open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args.add_argument(
        "-e",
        "--evaluate",
        default=False,
        type=bool,
        help="Calc argmax/beamsearch metrics on test",
    )

    args.add_argument(
        "--beam_size",
        default=100,
        type=int,
        help="Beam size for beam search",
    )

    args.add_argument(
        "--beam_search_type",
        default='torch',
        type=str,
        help="'torch' for pytorch bs, 'custom' for custom bs (slow but accurate), 'fast' for ctcdecode bs",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()
        config.config["data"] = {
            "test": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirAudioDataset",
                        "args": {
                            "audio_dir": str(test_data_folder / "audio"),
                            "transcription_dir": str(
                                test_data_folder / "transcriptions"
                            ),
                        },
                    }
                ],
            }
        }

    assert config.config.get("data", {}).get("test", None) is not None
    config["data"]["test"]["batch_size"] = args.batch_size
    config["data"]["test"]["n_jobs"] = args.jobs

    main(config, args.output, eval=args.evaluate,
         beam_size=args.beam_size, beam_search_type=args.beam_search_type)
