# SimpleFold patch — latent capture during sampling

`sampler.py` here is a **drop-in replacement** for
`ml-simplefold/src/simplefold/model/torch/sampler.py`.

The official `EMSampler.sample()` returns only the denoised coordinates. PROTEUS
needs the model's **trunk latent captured at a specific flow timestep** during the
sampling trajectory. This file adds exactly that and nothing else:

- `euler_maruyama_step(..., return_latent=False)` — optionally returns
  `out["latent"]` (the architecture already exposes this key);
- `sample(..., capture_at_flow_t=None)` — captures the latent at the sampler step
  closest to each requested flow-t and returns it in `captured_latents`.

It is byte-identical to the upstream sampler apart from these additions (the
Euler–Maruyama math is unchanged), so overlaying it onto the current official
release is safe. To apply:

```bash
cp simplefold_patch/sampler.py \
   <ml-simplefold>/src/simplefold/model/torch/sampler.py
```

The Colab notebook (`PROTEUS_colab/`) does this automatically after cloning
`apple/ml-simplefold`.
