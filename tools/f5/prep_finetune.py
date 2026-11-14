"""Turn an exported csv_wavs dir (scripts/tts_f5_export.py) into an F5-TTS fine-tune: arrow dataset, extended vocab,
checkpoint with matching text-embedding rows. Run with the F5 venv:

  tools/f5/.venv/bin/python tools/f5/prep_finetune.py data/tts/f5/multi_arabic --name brahui_multi_arabic

Writes F5-TTS/data/<name>_custom/{raw.arrow,duration.json,vocab.txt} and F5-TTS/ckpts/<name>/pretrained_model_1250000_ext.safetensors
and prints the finetune_cli command. New tokens get the mean of the pretrained embeddings plus small noise (random
normal rows, the upstream default, start far from the trained distribution).
"""
import argparse, sys
from pathlib import Path
import torch
from cached_path import cached_path
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent; F5 = HERE / "F5-TTS"
sys.path.insert(0, str(F5 / "src/f5_tts/train/datasets"))
from prepare_csv_wavs import prepare_csv_wavs_dir, save_prepped_dataset  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("--name", required=True)
a = ap.parse_args()
out = F5 / "data" / f"{a.name}_custom"
res, durs, vocab_set = prepare_csv_wavs_dir(str(Path(a.inp).resolve()), num_workers=4)
save_prepped_dataset(out, res, durs, vocab_set, True)  # copies the pretrained vocab
vocab = (out / "vocab.txt").read_text(encoding="utf-8").split("\n")
if vocab[-1] == "": vocab.pop()
have = set(vocab); miss = sorted(c for c in vocab_set if c not in have)
assert not any(c in "\n\r|" for c in miss), "metadata.csv was mis-parsed (stray quote?): newline or | among the new tokens"
(out / "vocab.txt").write_text("\n".join(vocab + miss) + "\n", encoding="utf-8")
print(f"vocab {len(vocab)} + {len(miss)} new: {' '.join(miss)}")

ck = Path(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"))
sd = load_file(str(ck), device="cpu"); key = "ema_model.transformer.text_embed.text_embed.weight"
old = sd[key]; print("pretrained text embedding", tuple(old.shape), "(vocab + 1 filler row)")
if miss:
    torch.manual_seed(0)
    new = old[1:len(vocab) + 1].mean(0, keepdim=True) + 0.02 * old.std() * torch.randn(len(miss), old.shape[1])
    sd[key] = torch.cat([old, new.to(old.dtype)], 0)
dst = F5 / "ckpts" / a.name; dst.mkdir(parents=True, exist_ok=True); dst = dst / "pretrained_model_1250000_ext.safetensors"
save_file(sd, str(dst))
print(f"\ncd {F5} && ../.venv/bin/accelerate launch src/f5_tts/train/finetune_cli.py --exp_name F5TTS_v1_Base --dataset_name {a.name} "
      f"--finetune --pretrain {dst} --tokenizer custom --tokenizer_path {out / 'vocab.txt'} "
      "--learning_rate 1e-5 --batch_size_per_gpu 3200 --epochs 100 --num_warmup_updates 2000 --save_per_updates 5000 --last_per_updates 1000 --keep_last_n_checkpoints 2")
