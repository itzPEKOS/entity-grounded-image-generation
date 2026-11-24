# Evaluation set

`prompts.csv` is the fixed prompt set. Do not change it between versions you want to compare.

- `split=dev` (15): look at these, find causes, fix. `part` A/B splits them between two Kaggle accounts.
- `split=holdout` (8): do NOT look at these while fixing. Run them once at the end to check the fixes
  were not tuned to the dev prompts.

Workflow
1. Kaggle: `kaggle/egig_eval.ipynb`, set `PART` (and the same `GIT_REF` for everyone), Save Version -> Save & Run All.
   Download `eval_<split>_<part>.zip` from the version's Output tab.
2. Laptop: unzip every part into one folder, e.g. `outputs/eval_dev/` (gitignored).
3. `python scripts/make_review.py outputs/eval_dev` -> open `review.html`, label each image, Export `labels.csv`.
   Label before reading any report, so the system's verdict does not steer you.
4. `python scripts/eval_report.py outputs/eval_dev --labels outputs/eval_dev/labels.csv`
5. Changed only the verifier? `python scripts/reverify.py outputs/eval_dev --out outputs/eval_dev_v2`
   then `python scripts/eval_report.py outputs/eval_dev --labels ... --system outputs/eval_dev_v2`
   (same images, no GPU). Changed the parser or generator? Run step 1 again.
