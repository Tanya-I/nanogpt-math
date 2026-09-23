# Fine-Tuning NanoGPT for Math with SFT and DPO

## What this project does

We start with a small pretrained GPT model (nanoGPT, `model.py`) that was originally
trained on general QA data and has no math ability. The goal of the lab is to fine-tune
it so it can solve basic arithmetic and algebra problems (e.g. `17+19=?`, `x*11=44,x=?`),
using two different fine-tuning approaches:

- **SFT (`sft/`)** — Supervised Fine-Tuning: train the model directly on
  prompt → correct-answer pairs.
- **DPO (`dpo/dpo.ipynb`)** — Direct Preference Optimization: train the model on pairs
  of a *bad* answer and a *good* answer for the same prompt, so it learns to prefer
  correct, well-explained solutions over wrong/unhelpful ones.

### How DPO fine-tuning works (`dpo/dpo.ipynb`)

1. **Load the pretrained checkpoint** (`gpt.pt`) — the base model before any math training.
2. **Load preference data** (`dpo/pos_neg_pairs.json`) — each entry has a `negative`
   response (incorrect/unhelpful, e.g. "Sorry, I don't know") and a `positive` response
   (the correct answer with reasoning). Text is cleaned to only characters the
   character-level tokenizer (`meta.pkl`) recognizes.
3. **Score both answers** — for each pair, the model computes the log-probability it
   assigns to the negative and the positive response (`compute_logprob`).
4. **DPO loss** — the training objective pushes the model to prefer the positive answer
   over the negative one:

   ```
   loss = -log(sigmoid(beta * (pos_logprob - neg_logprob))) - 0.1 * pos_logprob
   ```

   `beta` controls how strongly the model is pushed to prefer the correct answer; the
   extra `-0.1 * pos_logprob` term also directly encourages the correct answer's
   probability to go up.
5. **Train** — batches of pairs are run through the loop each epoch with AdamW +
   cosine-annealing learning rate schedule and mixed-precision (AMP) training; a
   checkpoint (`dpo.pt`) is saved after every epoch.
6. **Test** — the fine-tuned checkpoint is reloaded and prompted with arithmetic/algebra
   questions to check whether it now generates correct, explained answers instead of
   generic non-answers.

### Repo layout

| Path | Purpose |
|---|---|
| `model.py` | nanoGPT model definition (`GPT`, `GPTConfig`) shared by SFT and DPO |
| `configurator.py` | Simple CLI/config-file override helper (from nanoGPT) |
| `sft/` | Supervised fine-tuning data/checkpoints |
| `dpo/dpo.ipynb` | DPO fine-tuning notebook (data loading, training loop, testing) |
| `dpo/pos_neg_pairs.json` | Positive/negative response pairs used for DPO training |
| `TCMC_GROUP6_LAB.ipynb` | Group lab notebook |

## Contributors

- Grover Ekhnoor Kaur
- Irani Tanya
- Chan Zi Jian
