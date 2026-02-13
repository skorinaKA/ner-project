from datasets import load_dataset
from span_marker import SpanMarkerModel, SpanMarkerModelCardData, Trainer
from transformers import TrainingArguments
import evaluate
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from span_marker.evaluation import compute_f1_via_seqeval
import warnings
from typing import Dict

import torch
from sklearn.exceptions import UndefinedMetricWarning
from transformers import EvalPrediction

from span_marker.tokenizer import SpanMarkerTokenizer

def main():
    dataset_id = "DFKI-SLT/few-nerd"
    dataset = load_dataset(dataset_id, "supervised")
    labels = dataset["train"].features["ner_tags"].feature.names

    encoder_id = "bert-base-cased"
    model = SpanMarkerModel.from_pretrained(
        # Required arguments
        encoder_id,
        labels=labels,
        # Optional arguments
        model_max_length=256,
        entity_max_length=8,
        # To improve the generated model card
        model_card_data=SpanMarkerModelCardData(
            language=["en"],
            license="cc-by-sa-4.0",
            encoder_id=encoder_id,
            dataset_id=dataset_id,
        )
    )

    # metric = evaluate.load("accuracy")
    seqeval = evaluate.load("seqeval")
    
    # Metric helper method
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != IGNORE_INDEX]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [ [label_list[l] for l in label if l != IGNORE_INDEX] for label in labels ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)    
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


    # def compute_objective(metrics):
    #     return metrics["eval_accuracy"]

    args = TrainingArguments(
        output_dir="../models/span-marker-bert-base-fewnerd-coarse-super",
        report_to="wandb",
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
        fp16=True,
        warmup_ratio=0.1,
        dataloader_num_workers=2,
        run_name="spanmarker",
    )

    # args = TrainingArguments(
    #     output_dir='../models/span-marker-bert-base-fewnerd-coarse-super/checkpoint-final',
    #     report_to="wandb",
    #     logging_steps=5,
    #     per_device_train_batch_size=32,
    #     per_device_eval_batch_size=32,
    #     evaluation_strategy="steps",
    #     eval_steps=20,
    #     max_steps = 100,
    #     save_steps = 100
    # )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"].select(range(8000)),
        eval_dataset=dataset["validation"].select(range(2000)),
        compute_metrics=None,
    )
    trainer.train()

    # def optuna_hp_space(trial):
    #     return {
    #         "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
    #         "per_device_train_batch_size": trial.suggest_categorical(
    #             "per_device_train_batch_size", [16, 32, 64, 128]
    #         ),
    #     }    

    # best_run = trainer.hyperparameter_search(
    #     direction="maximize",
    #     backend="optuna",
    #     hp_space=optuna_hp_space,
    #     n_trials=5,
    #     compute_objective=compute_objective,
    # )
    # print(best_run)

    metrics = trainer.evaluate()
    print(metrics)

    trainer.save_model("../models/span-marker-bert-base-fewnerd-coarse-super/checkpoint-final")

if __name__ == "__main__":
    main()