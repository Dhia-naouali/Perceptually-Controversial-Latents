import os
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from .data import ImageDataset


def embed_images(
    model, 
    paths, 
    batch_size, 
    num_workers
):
    dataset = ImageDataset(paths)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    embeddings = []
    model.eval()

    with torch.no_grad():
        for images, _ in tqdm(loader):
            images = images.cuda(non_blocking=True)
            embs = model(images)
            embeddings.append(embs.cpu())

    return torch.cat(embeddings, dim=0)


def embed_optimized_images(
    model,
    optimized_images,
):
    model.eval()
    with torch.no_grad():

        return model(
            optimized_images.cuda(non_blocking=True)
        ).cpu()
    


def cosine_topk(query, gallery, k=2, method="max"):
    assert method in ["max", "mean"], f"topk similarity selected using mean or max got {method}"
    if query.ndim == 1:
        sims = gallery @ query
    else:
        sims = gallery @ query.T
        sims = sims.max(dim=1).values if method == "max" else sims.mean(dim=1)

    _, ids = sims.topk(min(k, gallery.size(0)))
    return ids

def retrieve(
    model,
    optimized_images,
    images_dir,
    topk,
    batch_size=256,
    num_workers=os.cpu_count() // 2,
    output_dir=None,
    top_sim_method="max"
):
    query_matrix = embed_optimized_images(model, optimized_images)
    query_centroids = F.normalize(query_matrix.mean(dim=0), dim=0)

    all_paths = list(Path(images_dir).glob("*.jpg")) +\
                list(Path(images_dir).glob("*JPEG")) +\
                list(Path(images_dir).glob("*.png"))
    
    natural_embeds = embed_images(
        model, 
        all_paths, 
        batch_size, 
        num_workers=num_workers
    )

    # mean query
    # meanq_ids = cosine_topk(query_centroids, natural_embeds, topk, method=top_sim_method)
    # all queries
    allq_ids = cosine_topk(query_matrix, natural_embeds, topk, method=top_sim_method)

    # union_ids = torch.cat([meanq_ids, allq_ids]).unique()
    union_ids = allq_ids
    union_sims = (natural_embeds[union_ids] @ query_matrix.T).mean(dim=1)
    rerank_order = union_sims.argsort(descending=True)

    retrieved_ids = union_ids[rerank_order[:topk]].tolist()
    retrieved_paths = [str(all_paths[i]) for i in retrieved_ids]
    retrieved_sims = union_sims[rerank_order[:topk]].tolist()

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        infos = {
            "image_ids": [Path(p).name for p in retrieved_paths],
            "image_paths": retrieved_paths,
            "sim_scores": retrieved_sims,
        }

        with open(output_dir / "retrieved_divergent_images.json", "w") as f:
            json.dump(infos, f, indent=2)

    return retrieved_paths, retrieved_sims
