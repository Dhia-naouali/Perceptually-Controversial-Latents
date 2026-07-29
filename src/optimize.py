import wandb
import math
import pandas as pd
from IPython import display
from pathlib import Path
from types import SimpleNamespace
from functools import partial

import torch
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm

from .losses import (
    divergence_loss, 
    ensemble_divergence_loss, 
    kl_divergence_loss, 
    annealed_ce_loss, 
    compute_mpcd
)
from .extractors import build_all_extractors
from .utils import MEANs, STDs, imagenet_prompts , save_images

STATS = (
    torch.tensor(MEANs).cuda().view(1, 3, 1, 1),
    torch.tensor(STDs).cuda().view(1, 3, 1, 1)
)

def unnormalize(images):
    m, s = STATS
    return (images.cuda().float() * s + m).clamp(0., 1.)

def _pixel_bounds():
    mean, std = STATS
    low, high = (0. - mean) / std, (1. - mean) / std
    return low, high

def _noise_init(b, c, h, w):
    low, high = _pixel_bounds()
    return torch.randn(b, c, h, w).cuda().mul(.5).clamp(low, high).detach().requires_grad_(True)


def _normalize_grads(x):
    if x.grad is None:
        return 
    
    b = x.shape[0]
    norms = x.grad.view(b, -1).norm(dim=1).clamp(min=1e-8)
    x.grad = x.grad / norms.view(b, 1, 1, 1)


def _log_images(run, name, images, step, config):
    save_every = config.save_every
    if step % save_every and step+1:
        return

    out_dir = config.out_dir
    images = unnormalize(
        images
    ).detach().permute(0, 2, 3, 1).cpu().numpy()
    
    if out_dir:
        image_file = f"step_{step}.png" if step+1 else "final_images.png"
        save_images(images, out_dir / image_file, title=f"{name} optimized images")


    if run is None:
        return 

    run.log({
        f"images_optimization/{name}/images": [
            wandb.Image(images[i]) for i in range(len(images))
        ]
    }, step=step)


def _log_comps(run, name, infos, step, config):
    log_every = config.log_every
    if run is None or step % log_every:
        return
    
    infos = {
        f"{name}/{k}": v for k, v in infos.items()
    }
    run.log(infos, step=step)



def _extract_config_for_optim(config, name):
    b, steps = config.optimization.batch_size, config.optimization.steps
    lr = config.optimization.lr
    lr_min = lr * config.optimization.lr_min_ratio
    h = w = config.mode.optimization.get("image_size", 224)
    repulsion_w = config.optimization.repulsion_weight
    log_every = config.optimization.log_every
    save_every = config.optimization.save_every

    out_dir = Path(config.output.dir) / name if config.output.dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    return SimpleNamespace(**{
        "b": b, "steps": steps, "lr": lr, "lr_min": lr_min,
        "h": h, "w": w, "repulsion_w": repulsion_w, 
        "log_every": log_every, "save_every": save_every,
        "out_dir": out_dir
    })
    

def _optimize_pixels_ensemble(config, extractor, run):
    name = "pixels_ensemble"
    c = _extract_config_for_optim(config, name)
    weights = {
        m.name: m.weight for m in config.extractor.members
    } if hasattr(config.extractor.members[0], "weight") else None

    images = _noise_init(c.b, 3, c.h, c.w)
    low, high = _pixel_bounds()
    optimizer = optim.Adam([images], lr=c.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c.steps, eta_min=c.lr_min)

    for step in (pb := tqdm(range(c.steps))):
        optimizer.zero_grad()
        feats = extractor(images) # dict
        loss, comps = ensemble_divergence_loss(feats, weights, c.repulsion_w)
        loss.backward()
        _normalize_grads(images)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            images.data.clamp_(low, high)
        pb.set_postfix({f"{name}/loss": loss.item()})
        
        _log_comps(run, name, comps, step, c)
        _log_images(run, name, images, step, c)


    _log_images(run, name, images, -1, c)
    return images.detach().cpu()


def _optimize_pixels_clip(config, extractor, run):
    name = "pixels_clip"
    c = _extract_config_for_optim(config, name)
    images = _noise_init(c.b, 3, c.h, c.w)
    low, high = _pixel_bounds()
    optimizer = optim.Adam([images], lr=c.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c.steps, eta_min=c.lr_min)

    for step in (pb := tqdm(range(c.steps))):
        optimizer.zero_grad()
        feats = extractor(images)
        loss, comps = divergence_loss(feats, c.repulsion_w)
        loss.backward()
        _normalize_grads(images)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            images.data.clamp_(low, high)
        pb.set_postfix({f"{name}/loss": loss.item()})
        
        _log_comps(run, name, comps, step, c)
        _log_images(run, name, images, step, c)


    _log_images(run, name, images, -1, c)
    return images.detach().cpu()


def _opitmizer_pixels_kl(config, extractor, run):
    name = "pixel_kl"
    c = _extract_config_for_optim(config, name)
    kl_config = config.mode.kl

    stride = 1000 // c.b
    targets = torch.tensor([(i * stride) % 1000 for i in range(c.b)], dtype=torch.long).cuda()
    images = _noise_init(c.b, 3, c.h, c.w)
    low, high = _pixel_bounds()
    optimizer = optim.Adam([images], lr=c.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c.steps, eta_min=c.lr_min)

    for step in (pb := tqdm(range(c.steps))):
        optimizer.zero_grad()
        logits = extractor(images)
        kl_loss, kl_comps = kl_divergence_loss(logits)
        ce_loss, ce_w = annealed_ce_loss(
            logits, 
            targets, 
            step, 
            c.steps, 
            kl_config.target_anneal_frac, 
            kl_config.target_weight
        )
        loss = kl_loss + ce_loss
        loss.backward()
        _normalize_grads(images)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            images.data.clamp_(low, high)

        comps = {**kl_comps, "ce_w": ce_w, "total_loss": loss.item()}
        pb.set_postfix({f"{name}/loss": loss.item()})

        _log_comps(run, name, comps, step, c)
        _log_images(run, name, images, step, c)


    _log_images(run, name, images, -1, c)
    return images.detach().cpu()



def _optimize_flux(config, extractor, flux, run):
    name = "flux"
    c = _extract_config_for_optim(config, name)

    flux_config = config.mode.flux
    if config.extractor.name == "ensemble":
        weights = {
            m.name: m.weight for m in config.extractor.members
        } if hasattr(config.extractor.members[0], "weight") else None
        loss_func = partial(ensemble_divergence_loss, weights=weights)
    else: 
        loss_func = divergence_loss

    prompts = [
        imagenet_prompts[i % len(imagenet_prompts)] for i in range(c.b)
    ]

    z, t5_embeds, clip_embeds = flux.init_latents(c.b, seed=flux_config.latents_seed, prompts=prompts)
    param_groups = []
    if flux_config.optimize_z:
        param_groups.append({
            "params": [z], 
            "lr": flux_config.lr_z
        })
    if flux_config.optimize_c:
        param_groups.append({
            "params": [t5_embeds, clip_embeds], 
            "lr": flux_config.lr_c
        })
    optimizer = optim.Adam(param_groups)

    for step in (pb := tqdm(range(c.steps))):
        optimizer.zero_grad()
        images_norm, latents = flux.decode(z, t5_embeds, clip_embeds)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            scale, shift = flux.vae.config.scaling_factor, flux.vae.config.shift_factor
            scaled_latents = (latents / scale) + shift
            images_grad = flux.vae.decode(scaled_latents, return_dict=False)[0]
            images_grad = flux._clamp_norm_vae(images_grad)

        feats = extractor(images_grad)
        loss, comps = loss_func(feats, repulsion_weight=c.repulsion_w)
        loss.backward()

        all_params = [t for g in param_groups for t in g["params"]]
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.)
        optimizer.step()

        for i, param_g in enumerate(optimizer.param_groups):
            base = flux_config.lr_c if i else flux_config.lr_z
            param_g["lr"] = c.lr_min + .5 * (base - c.lr_min) * (
                1. + math.cos(math.pi * step / max(1, c.steps))
            )

        with torch.no_grad():
            flux.clamp_latents(z)
        
        pb.set_postfix({f"{name}/loss": loss.item()})

        _log_comps(run, name, comps, step, c)
        _log_images(run, name, images_norm, step, c)

    with torch.no_grad():
        images = flux.decode(z, t5_embeds, clip_embeds)[0].detach().cpu()

    _log_images(run, name, images, -1, c)
    return images


@torch.no_grad()
def cross_evaluate(images):
    results = {}
    for extractor_name, extractor in build_all_extractors().items():
        images = images.cuda()
        if extractor_name == "classifier":
            feature_map = extractor.model.forward_features(images)
            feats = extractor.model.forward_head(
                feature_map,
                pre_logits=True,
            )
        else:
            feats = extractor(images)
        mpcd = compute_mpcd(feats)
        results[extractor_name] = mpcd

    return results

def optimize_images(config, extractor, run=None, generator=None):
    mode = config.mode.name

    if mode == "pixels_ensemble":
        images = _optimize_pixels_ensemble(config, extractor, run=run)
    elif mode == "pixels_clip":
        images = _optimize_pixels_clip(config, extractor, run=run)
    elif mode == "pixels_kl":
        images = _opitmizer_pixels_kl(config, extractor, run=run)
    elif mode == "flux":
        images = _optimize_flux(config, extractor, generator, run=run)
    else:
        raise ValueError(f"unkonwn mode: {mode}")
    
    cross_eval = {}
    if config.cross_eval.enabled:
        cross_eval = cross_evaluate(images)
        print(cross_eval)
    
    return images, cross_eval
