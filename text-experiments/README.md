# project-scalion

## Text Experiments

# DPO Pipeline

To train a full pipeline that involves naive paraphrasing, scoring and preference data generation.

`bash scripts/baseline_pipeline.sh $TASK $PARAPHASER_MODEL $EVALUATOR_MODEL $TRAINED_MODEL_PATH`

```
bash scripts/baseline_pipeline.sh \
    gsm_symbolic \
    Qwen/Qwen2.5-3B-Instruct \
    Qwen/Qwen2.5-7B-Instruct \
    ${OUTPUT_PATH}$/dpo-qwen2.5-3b-from-qwen2.5-7b/
```