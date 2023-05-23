# peft-debias
Debiasing language models using parameter efficient fine tuning techniques

This code requires `adapter-transformers` library.

```bash
pip install adapter-transformers
```

Running Upstream Intrinsic Experiments
```bash
bash run_upstream.sh pfeiffer 0 cda gender bias-bios
```

It requires peft (pfeiffer), gpu id (0), debias technique (cda), dataset (bias-bios)

Running Upstream Intrinsic Scores
```bash
bash run_upstream_scores.sh cda gender crows sft gender
```

Running Downstream Experiments
```bash
bash run_downstream.sh pfeiffer bias-bios $PATCH_PATH p cda
```
where $PATCH_PATH is the path of the patch/peft that we obtain during upstream.

Running Downstream Extrinsic Scores
```bash
bash run_downstream_scores.sh pfeiffer bias-bios cda gender
```