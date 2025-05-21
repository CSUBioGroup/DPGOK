import argparse
import pathlib

import torch

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer

import pickle as pkl

def extract_esm(fasta_file, model_location,
                truncation_seq_length=1022, toks_per_batch=4096,
                device='cuda', out_file=None):
    
    model, alphabet = pretrained.load_model_and_alphabet(model_location)
    model.eval()
    if device:
        model = model.to(device)
    
    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length), batch_sampler=batches
    )
    print(f"Read {fasta_file} with {len(dataset)} sequences")

    # output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = False
    
    repr_layers = [36,]
    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]
    print(repr_layers)
    
    pre_dict = {}
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if device:
                toks = toks.to(device, non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)

            logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            if return_contacts:
                contacts = out["contacts"].to(device="cpu")

            for i, label in enumerate(labels):
                result = {"label": label}
                truncate_len = min(truncation_seq_length, len(strs[i]))
                result["mean_representations"] = {
                    layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                    for layer, t in representations.items()
                }
                pre_dict[label] = result["mean_representations"][36]
    print(pre_dict)
    if out_file is not None:
        with open(out_file, 'wb') as fw:
            pkl.dump(pre_dict, fw)
    return pre_dict

def main(seq_file, output, esm_model):
    extract_esm(fasta_file=seq_file, model_location=esm_model, out_file=output)


if __name__ == '__main__':
    for t in ['test', 'valid', 'train']:
        seq_file = f'../dataset/{t}_seq.fasta'
        output = f'../dataset/{t}_esm2.pkl'
        main(seq_file,output,'../esm2/esm2_t36_3B_UR50D.pt')
