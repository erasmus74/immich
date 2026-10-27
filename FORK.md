# Fork runbook

Fork of [immich-app/immich](https://github.com/immich-app/immich) (AGPL-3.0). Fork additions live on `fork/ml-nl-search`; `main` stays a pristine mirror of upstream.

## Layout

| Branch | Content |
|---|---|
| `main` | upstream mirror — fast-forward only, never commit here |
| `fork/ml-nl-search` | VLM captioning pipeline + NL search improvements |

## Update from upstream

```bash
git checkout main
git fetch upstream --tags
git merge --ff-only upstream/main   # fails loudly if diverged — do not force
git push origin main
git checkout fork/ml-nl-search
git rebase main                     # resolve, then push --force-with-lease to origin fork branch
```

Known rebase hotspots (heavily-churned files this fork touches): `server/src/dtos/config.dto.ts`, `web/src/routes/admin/system-settings/MachineLearningSettings.svelte`, `machine-learning/immich_ml/pipeline.py`, `machine-learning/immich_ml/models/constants.py`.

---

## Feature 1: external ML on LAN GPU box, host fallback (config-only — no fork code)

Upstream already routes every ML request across a URL list: healthy servers first, unhealthy as fallback, with a background health pinger. Setup:

### 1. GPU box (`gpu-box.lan`)

Run the standard ML container. Example compose fragment:

```yaml
services:
  immich-machine-learning:
    container_name: immich-machine-learning-gpu
    image: ghcr.io/immich-app/immich-machine-learning:<same version as server>
    restart: unless-stopped
    ports: ['3003:3003']
    volumes:
      - model-cache:/cache
    environment:
      MACHINE_LEARNING_DISABLED_MODELS: ''   # GPU box runs everything, incl. heavy caption models
    gpus: all                          # or runtime: nvidia + NVIDIA_VISIBLE_DEVICES
volumes:
  model-cache:
```

### 2. Immich host (fallback hardware)

The normal `immich-machine-learning` service from the stock compose stays as-is. Restrict it from inheriting the GPU box's heavy jobs:

```yaml
    environment:
      MACHINE_LEARNING_DISABLED_MODELS: 'fancyfeast/*,unsloth/*'  # example: deny-list prefix globs
```

Unset/empty = allow all (stock behavior).

### 3. Point the server at both

Admin UI → Administration → Settings → Machine Learning Settings:

1. **URL 1:** `http://gpu-box.lan:3003` (order matters — first = preferred)
2. **URL 2:** `http://immich-machine-learning:3003` (local fallback)
3. **Availability checks:** enabled (default: 2 s timeout, 30 s interval)

Verify: server logs show `Machine learning server became healthy (<url>)` per URL. Kill the GPU box container and watch the second URL take over within one health interval; jobs requeue and complete on local hardware.

Notes:
- Both instances must be able to load the models your config references; a model an instance refuses returns 422 and the server falls through to the next URL on that same request.
- Model downloads cache per instance (`/cache` volume) — pre-warm by running a job once.

---

## Feature 2: VLM captions + NL search (`fork/ml-nl-search`)

### What it adds

1. **Captioning task** in the ML service with two engines:
   - `Qwen/Qwen3-VL-2B-Instruct` — native transformers VLM, CPU-viable, richly descriptive (fallback engine). `microsoft/Florence-2-base` is NOT usable: its remote code requires transformers 4.x, which conflicts with the service's `huggingface-hub>=1.0` pin — the engine raises a clear error.
   - `fancyfeast/joy-caption-alpha-two` — candid descriptions incl. clothing/pose/setting detail (primary engine; needs a real GPU, ~8 GB+ VRAM with 4-bit quant).
2. **Hybrid routing** via existing multi-URL failover: GPU box instance runs JoyCaption; local instance deny-lists it (`MACHINE_LEARNING_DISABLED_MODELS`) and serves Qwen3-VL. Primary-model failure automatically retries the fallback model.
3. **Search integration:**
   - Captions are tokenized into full-text search (`asset_caption.searchText`) and blended into the smart-search embedding: `normalize((1-w)·img + w·txt)`, `w = embeddingWeight` (default 0.3, 0 = off).
   - Person names in queries are matched against your people and turned into person filters, with the remainder embedded: *"pictures of alice wearing red lingerie fall boudoir photoshoot"* → `personIds=[Alice]` + CLIP over *"wearing red lingerie fall boudoir photoshoot"* + FTS over captions. Requires the person to exist in People with ≥3-char name.
4. **Better base model (recommended):** switch Smart Search CLIP model to `ViT-SO400M-14-SigLIP2__webli` in admin settings — significantly stronger retrieval than the default `ViT-B-32__openai`.

### Deployment

Build the ML image with the captioning extra (default image is unchanged and captionless):

```bash
docker build -f machine-learning/Dockerfile \
  --build-arg INSTALL_CAPTION=true \
  -t immich-ml-captioned:<ver> machine-learning
```

Run `immich-ml-captioned` on the GPU box and the same captioned image locally with the JoyCaption deny-list (Qwen3-VL runs on CPU). The stock upstream image has no captioning engines — caption requests return an error there.

### Admin config (Machine Learning Settings → Captioning)

| Setting | Value | Notes |
|---|---|---|
| Enabled | on | |
| Model | `fancyfeast/joy-caption-alpha-two` | primary |
| Fallback model | `Qwen/Qwen3-VL-2B-Instruct` | used if all URLs refuse/fail primary |
| Prompt | empty | engine default; override to steer captions |
| Max tokens | 120 | caption length cap |
| Embedding weight | 0.3 | 0 disables blend |

### After changing CLIP model or enabling blend

1. Administration → Jobs → **Smart Search** → All (re-embed; dimension change is migrated automatically).
2. Administration → Jobs → run the **Captioning** job over the library.
3. Re-run Smart Search again after captions complete if blending was on and captions are new (embeddings change once captions exist).

### First-run costs

- JoyCaption pulls ~16 GB (vision tower + adapter + LLM) on first use per instance; Qwen3-VL-2B ~4 GB.
- Captioning throughput: ~1–3 img/s on a mid-range GPU (JoyCaption). CPU with Qwen3-VL-2B measured ~30 s per 120-token caption on 16 threads — use the GPU box as primary.

## License

AGPL-3.0 (inherited). Keep the fork public or provide source to service users; rebrand before public exposure (Immich name/logo are trademarks, not covered by the license).
