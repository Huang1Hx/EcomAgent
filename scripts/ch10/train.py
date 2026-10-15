"""ch10 训练:RoBERTa-wwm-ext 全参微调,17 类多标签(BCEWithLogitsLoss)。
运行:make ch10-train。设备自适应 cuda→mps→cpu(Trainer 自动挑);正则化 weight_decay + 早停盯验证集 micro-F1。
HF 下载不通时:HF_ENDPOINT=https://hf-mirror.com make ch10-train。"""
import json
import pathlib
import argparse
import time

import numpy as np
from sklearn.metrics import f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EarlyStoppingCallback, Trainer, TrainerCallback,
                          TrainingArguments, default_data_collator, set_seed)

from app.core.taxonomy import ID2LABEL, LABEL2ID, NUM_CLASSES
from scripts.ch10 import artifacts
from scripts.ch10.inference_lib import apply_threshold

BASE = "hfl/chinese-roberta-wwm-ext"
DATA = artifacts.DATA
OUT = artifacts.MODEL


def load_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def encode(samples: list[dict], tokenizer) -> list[dict]:
    enc = tokenizer([s["text"] for s in samples], truncation=True,
                    padding="max_length", max_length=128)
    items = []
    for i, s in enumerate(samples):
        vec = [0.0] * NUM_CLASSES                 # float 向量:BCEWithLogitsLoss 要求
        for lb in s["labels"]:
            vec[LABEL2ID[lb]] = 1.0
        items.append({"input_ids": enc["input_ids"][i],
                      "attention_mask": enc["attention_mask"][i],
                      "token_type_ids": enc["token_type_ids"][i],
                      "labels": vec})
    return items


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = apply_threshold(1 / (1 + np.exp(-logits)), 0.5)
    return {"micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0)}


class BestInMemory(TrainerCallback):
    """最优权重快照进内存,全程零磁盘 checkpoint。
    本机磁盘只剩几百 MB:Trainer 的 save/rotate 峰值要 2-3 份权重(实测两次撞满盘),
    改为 on_evaluate 时把更优的 state_dict 复制到 CPU 内存,训练完回填、只落一次盘。"""

    def __init__(self, model):
        self.model = model
        self.best_metric = -1.0
        self.best_state = None
        self.best_epoch = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        score = (metrics or {}).get("eval_micro_f1", -1.0)
        if score > self.best_metric:
            self.best_metric = score
            self.best_epoch = state.epoch
            self.best_state = {k: v.detach().to("cpu", copy=True)
                               for k, v in self.model.state_dict().items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=pathlib.Path, default=DATA)
    ap.add_argument("--base-model", default=BASE)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    cli = ap.parse_args()
    frozen = artifacts.dataset_manifest(cli.data_dir)
    artifacts.write_json(artifacts.CH10 / "dataset-manifest.json", frozen)
    if (OUT / "model.safetensors").exists():
        raise SystemExit(f"模型已存在，请先归档再训练: {OUT}")
    started = time.monotonic()
    meta = artifacts.provenance()
    set_seed(cli.seed)  # 分类头随机初始化之前设定种子
    tokenizer = AutoTokenizer.from_pretrained(cli.base_model, revision=cli.revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        cli.base_model, revision=cli.revision, num_labels=NUM_CLASSES, problem_type="multi_label_classification",
        id2label=ID2LABEL, label2id=LABEL2ID)
    train_ds = encode(load_jsonl(cli.data_dir / "train.jsonl"), tokenizer)
    val_ds = encode(load_jsonl(cli.data_dir / "val.jsonl"), tokenizer)
    args = TrainingArguments(
        output_dir=str(artifacts.CH10 / "checkpoints"),
        eval_strategy="epoch",                    # v5 参数名,不是 evaluation_strategy
        save_strategy="no",                       # 零磁盘 checkpoint,最优权重走 BestInMemory
        learning_rate=2e-5,
        per_device_train_batch_size=cli.batch_size,
        per_device_eval_batch_size=32,
        num_train_epochs=cli.epochs,
        seed=cli.seed,
        data_seed=cli.seed,
        disable_tqdm=True,
        dataloader_pin_memory=False,
        weight_decay=0.01,                        # 正则化防过拟合
        metric_for_best_model="micro_f1",         # EarlyStopping 盯它
        greater_is_better=True,
        logging_steps=20,
        report_to="none",
    )
    best_cb = BestInMemory(model)
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tokenizer,               # v5:tokenizer= 已改名
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2), best_cb],
    )
    trainer.train()
    if best_cb.best_state is not None:            # 回填验证集最优权重(替代 load_best_model_at_end)
        model.load_state_dict(best_cb.best_state)
        print(f"回填最优权重:验证集 micro-F1 {best_cb.best_metric:.4f}")
    # 验证集上扫全局最优阈值(0.30~0.70 步进 0.05)
    logits = trainer.predict(val_ds).predictions
    probs = 1 / (1 + np.exp(-logits))
    gold = np.array([d["labels"] for d in val_ds])
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.30, 0.71, 0.05):
        f1 = f1_score(gold, apply_threshold(probs, round(float(t), 2)), average="micro", zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = round(float(t), 2), float(f1)
    OUT.mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUT)
    tokenizer.save_pretrained(OUT)
    artifacts.verify_dataset(frozen, cli.data_dir)
    artifacts.write_json(OUT / "threshold.json", {"threshold": best_t, "val_micro_f1": best_f1})
    report = {**meta, "dataset": frozen, "base_model": BASE,
              "base_revision": getattr(model.config, "_commit_hash", None) or pathlib.Path(cli.base_model).name,
              "train_config": {"seed": cli.seed, "batch_size": cli.batch_size,
                               "max_epochs": cli.epochs, "learning_rate": 2e-5,
                               "weight_decay": 0.01, "max_length": 128, "patience": 2},
              "device": str(args.device), "elapsed_seconds": round(time.monotonic() - started, 2),
              "best_epoch": best_cb.best_epoch, "best_val_micro_f1_at_05": best_cb.best_metric,
              "threshold": best_t, "val_micro_f1": best_f1, "history": trainer.state.log_history}
    artifacts.write_json(artifacts.REPORTS / "training_report.json", report)
    artifacts.write_json(OUT / "artifact-manifest.json", {
        "labels": list(artifacts.TOPIC_NAMES), "dataset": frozen,
        "training": report, "files": artifacts.file_hashes(OUT)})
    print(f"最优阈值 {best_t},验证集 micro-F1 {best_f1:.4f};模型已存 {OUT}")


if __name__ == "__main__":
    main()
