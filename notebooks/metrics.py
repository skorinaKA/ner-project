import wandb
from datasets import load_dataset
from span_marker import SpanMarkerModel, SpanMarkerModelCardData, Trainer
from transformers import TrainingArguments

def main():
    # 1. Инициализация wandb
    wandb.init(project="spanmarker-fewnerd", name="spanmarker-bert-base-cased")

    # 2. Загрузка датасета FewNERD (coarse-grained)
    dataset_id = "DFKI-SLT/few-nerd"
    dataset = load_dataset(dataset_id, "supervised")
    labels = dataset["train"].features["ner_tags"].feature.names

    # 3. Загрузка предобученной модели SpanMarker
    encoder_id = "bert-base-cased"
    model = SpanMarkerModel.from_pretrained(
        encoder_id,
        labels=labels,
        model_max_length=256,
        entity_max_length=8,
        model_card_data=SpanMarkerModelCardData(
            language=["en"],
            license="cc-by-sa-4.0",
            encoder_id=encoder_id,
            dataset_id=dataset_id,
        )
    )

    # 4. Аргументы обучения
    args = TrainingArguments(
        output_dir="../models/span-marker-bert-base-fewnerd-coarse-super",
        report_to="wandb",                     # автоматическое логирование в wandb
        learning_rate=5e-5,
        gradient_accumulation_steps=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        push_to_hub=False,
        logging_steps=50,
        fp16=True,                            # если GPU поддерживает
        warmup_ratio=0.1,
        dataloader_num_workers=2,
        run_name="spanmarker-fewnerd",
    )

    # 5. Trainer (из SpanMarker) – compute_metrics не передаём,
    #    поэтому используется встроенная оценка через seqeval
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"].select(range(8000)),
        eval_dataset=dataset["validation"].select(range(2000)),
        compute_metrics=None,   # ВАЖНО: не указываем свою функцию
    )

    # 6. Обучение
    trainer.train()

    # 7. Финальная оценка и логирование метрик в wandb
    metrics = trainer.evaluate()
    print(metrics)
    wandb.log(metrics)   # финальные метрики также сохраняются

    # 8. Сохранение модели
    trainer.save_model("../models/span-marker-bert-base-fewnerd-coarse-super/checkpoint-final")

    # 9. Сохранение модели как артефакта wandb (опционально)
    best_model_path = trainer.state.best_model_checkpoint
    if best_model_path:
        artifact = wandb.Artifact("spanmarker-fewnerd-model", type="model")
        artifact.add_dir(best_model_path)
        wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == "__main__":
    main()