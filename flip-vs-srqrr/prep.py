"""Prepare byte-level WikiText-2 (pytorch/examples files, = teammate CPU runs) and WikiText-103 for qtrain.py."""
import os, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qtrain as Q
W = os.path.dirname(os.path.abspath(__file__))
D2 = os.path.join(W, "data", "wt2_pytorch")
if not os.path.exists(D2 + "/meta.json"):
    base = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/"
    T = {s: urllib.request.urlopen(base + f + ".txt").read().decode("utf-8") for s, f in (("train", "train"), ("val", "valid"), ("test", "test"))}
    Q.prepare_from_texts(D2, T)
D103 = os.path.join(W, "data", "wt103")
if not os.path.exists(D103 + "/meta.json"):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    Q.prepare_from_texts(D103, {s: "".join(ds[h]["text"]) for s, h in (("train", "train"), ("val", "validation"), ("test", "test"))})
print("DATA READY")
