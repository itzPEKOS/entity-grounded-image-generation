# Entity-Grounded Image Generation

A four-agent system that splits a text-to-image request into entities, looks up what each
entity really looks like (facts and real reference photos), uses those photos to steer an
SDXL diffusion model, and verifies the result with a detector and a vision-language model.
Every check in the report is marked PASS or FAIL. If the image fails, the failed checks go
back to the generator, which adjusts its settings and tries again.

```
user prompt
  └─► Agent 1  Query Parser       Gemini → constraint spec (entities, layout boxes, scene, relations)
        └─► Agent 2  Entity Retriever  Wikipedia + Commons photos → curated with Grounding DINO + CLIP → visual profile (Gemini)
              └─► Agent 3  Image Generator  SDXL + IP-Adapter (look, masked per entity) + ControlNet canny (structure of rigid entities)
                    └─► Agent 4  Verifier       Grounding DINO (presence / count / position) + Gemini (identity vs. reference, attributes, features, scene, relations)
                          ├─ PASS → final image + report
                          └─ FAIL → failed checks + fix hints → Agent 3 repairs its knobs → next attempt (max 3)
```

## Where things run

| Part | Model | Runs on |
|---|---|---|
| Agents 1, 2 (profile), 4 (VLM checks) | `gemini-3.5-flash-lite` (free tier) | Google's servers, via API |
| Agent 2 retrieval | Wikipedia / Wikimedia Commons API | Wikimedia's servers |
| Agent 2 curation, Agent 4 detection | Grounding DINO tiny + CLIP ViT-B/32 | GPU 1 on Kaggle (CPU also works) |
| Agent 3 | SDXL base + IP-Adapter Plus (ViT-H) + ControlNet canny | GPU 0 on Kaggle (T4 16 GB, with CPU offload) |

Without a CUDA GPU (e.g. a laptop), Agent 3 automatically uses a **collage stub**. The stub pastes the reference photos into the planned boxes so the rest of the loop can be tested. It is not diffusion.

Gemini calls per run = 1 (parse) + 1 per looked-up entity + 1 per attempt. Example: 2 entities and 3 attempts = 1 + 2 + 3 = 6 calls.

## Setup

You need two values:

| Variable | Why | Where to get it |
|---|---|---|
| `GEMINI_API_KEY` | Agents 1, 2, 4 | https://aistudio.google.com/apikey (free) |
| `WIKI_CONTACT` | Wikimedia refuses requests (HTTP 403) without contact info in the User-Agent | your email or a URL, e.g. your GitHub profile |

### Laptop (logic + Gemini + Wikipedia; collage instead of SDXL)

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# optional: detector + CLIP on CPU
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install "transformers>=4.45"
copy .env.example .env            # then fill in GEMINI_API_KEY and WIKI_CONTACT
python -m egig doctor
python -m egig run --prompt "A Shiba Inu sitting in front of Wat Arun at sunset"
```

### Kaggle (full system on GPU T4 ×2)

1. `python scripts/make_kaggle_zip.py`, then upload `dist/egig_code.zip` as a Kaggle Dataset.
2. Create a notebook from `kaggle/egig_kaggle.ipynb`. Under Session options, set the accelerator to **GPU T4 x2** and turn **Internet** on. Add the dataset as an input.
3. Add `GEMINI_API_KEY` and `WIKI_CONTACT` under Add-ons > Secrets.
4. Run the cells in order. Run `python -m egig smoke` once first: it loads every GPU model and makes test images.

## Commands

```bash
python -m egig run --prompt "..." [--max-attempts 3] [--generator auto|sdxl|collage] [--detector auto|gdino|none] [--size 1024] [--steps 25]
python -m egig doctor     # key, models your key can use, Wikipedia access, GPU
python -m egig smoke      # GPU: SDXL, masked IP-Adapter, Grounding DINO
pytest                    # 36 tests: fakes for all external services + a tiny random SDXL on CPU
```

Every field in `egig/config.py` can be overridden with an environment variable of the same name in UPPERCASE, e.g. `MAX_ATTEMPTS=4`.

## Output of a run (`outputs/<timestamp>/`)

| File | Content |
|---|---|
| `01_spec.json` | Agent 1 output: entities, layout boxes, scene requirements, relations |
| `02_spec_grounded.json` | + reference photos (with detector/CLIP scores), visual profiles, source URLs |
| `03_checks.json` | the checklist Agent 4 will verify |
| `refs/<entity>/` | downloaded photos and the crops used as references |
| `attempt_N/` | `image.png`, `prompt.txt`, `layout.png`, `control_canny.png`, `detections.png`, `report.md/json` |
| `final.png`, `final_report.md` | best attempt + summary table of all attempts |
| `trace.jsonl` | every message between agents (who sent what to whom) |

## How PASS/FAIL is decided

Checks are either **critical** or **major**:
- **Critical:** entity present, count, identity matches the reference, user-requested attributes, scene requirements, relations.
- **Major:** position matches the layout, key visual features.

**PASS = every critical check passes AND at least 70 % of major checks pass.** The verdict is computed in code (`agents/verifier.py: compute_verdict`) and never taken from the VLM. If the VLM skips a check, that check counts as FAIL. If no attempt passes, the run returns the attempt that passed the most critical checks.

## Feedback loop (`egig/repair.py`)

| Failed check | What Agent 3 changes next attempt |
|---|---|
| presence | entity first in prompt, IP-Adapter +0.10 |
| identity | IP-Adapter +0.15 (closer to the reference), add the entity's negative traits |
| user attribute (e.g. "sitting") | IP-Adapter −0.15 (a strong reference copies the photo's pose), entity first in prompt |
| scene (e.g. "at sunset") | every IP-Adapter −0.05 (reference photos carry their own lighting) |
| position of a rigid entity | ControlNet +0.10 |
| any | new seed, VLM fix hint added to the prompt |

Every change is written to the attempt's report together with the reason for it.
