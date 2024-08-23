import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List

from torchinfo import summary 

import torch
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)


from sklearn.metrics import mean_squared_error
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd 

torch.backends.cudnn.enabled = False  

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps"),
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)




def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


"""
Get the reversed complement of the original DNA sequence.
"""
def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # return "".join([MAP[c] for c in reversed(sequence)])
    return "".join([MAP[c] for c in sequence])

"""
Transform a dna sequence to k-mer string
"""
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


"""
Load or generate k-mer string for each DNA sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""
def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:        
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
        
    return kmer

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 kmer: int = -1):

        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        try : 
            
            if len(data[0]) >= 2:
                # data is in the format of [text, label]
                logging.warning("Perform single sequence classification...")
                texts = [d[0] for d in data]
                labels = [d[1] for d in data]
            elif len(data[0]) == 3:
                # data is in the format of [text1, text2, label]
                logging.warning("Perform sequence-pair classification...")
                texts = [[d[0], d[1]] for d in data]
                labels = [int(d[2]) for d in data]
            else:
                raise ValueError("Data format not supported.")
        except : 
            pass
            
        if kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)

            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = 1

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        import decimal

        # Assuming instances is a sequence of dictionaries containing "labels" as strings
        # Convert labels to decimal numbers
        for instance in instances:
            instance["labels"] = decimal.Decimal(instance["labels"])
        
        # Extract input_ids and labels from instances
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]

        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # print(f"2 : {labels}")
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.Tensor(labels).float()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])
    predictions = np.argmax(logits, axis=-1)
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }


"""
Compute metrics used for huggingface trainer.
""" 
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     if isinstance(logits, tuple):  # Unpack logits if it's a tuple
#         logits = logits[0]
#     return calculate_metric_with_sklearn(logits, labels)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Flatten predictions and labels arrays
    predictions = predictions[0].flatten()
    labels = labels.flatten()
    
    # Calculate correlation coefficient between predictions and labels
    correlation_coefficient = np.corrcoef(predictions, labels)[0, 1]
    
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(labels, predictions)
    
    return {"correlation_coefficient": correlation_coefficient, "mse": mse}






def train():
    torch.cuda.empty_cache()

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token

    # define datasets and data collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                      data_path=os.path.join(data_args.data_path, "train.csv"), 
                                      kmer=data_args.kmer)
    val_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "dev.csv"), 
                                     kmer=data_args.kmer)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)


    # load model
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        num_labels=train_dataset.num_labels,
        trust_remote_code=True,
    )

    summary(model)

    # configure LoRA
    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # define trainer
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   data_collator=data_collator,
                                   
                                  )
    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # get the evaluation results from trainer
    if training_args.eval_and_save_results:

        # Load the CSV file
        file_path = os.path.join(data_args.data_path, "test.csv")  # Replace with your actual file path
        df = pd.read_csv(file_path)

        num_rows = len(df)
         # Calculate the size of each partition
        partition_size = num_rows // 7

        # Split the DataFrame into 7 parts
        partitions = []
        for i in range(6):
            start_idx = i * partition_size
            end_idx = (i + 1) * partition_size
            partitions.append(df.iloc[start_idx:end_idx])

        # The last partition contains the remaining rows
        partitions.append(df.iloc[6 * partition_size:])
        
        results_path = os.path.join(training_args.output_dir, "results")
        os.makedirs(results_path, exist_ok=True)
        # Save each partition as a separate CSV file
        for i, partition in enumerate(partitions):
            partition_path = os.path.join(results_path, f"partition_{i+1}.csv")
            partition.to_csv(partition_path, index=False)
            test_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                         data_path=partition_path, 
                                         kmer=-1)

            # get the evaluation results from trainer
            results = trainer.predict(test_dataset)
            predictions = results[0][0].reshape((len(results[0][0]), 1))
            ground_truth = pd.read_csv(partition_path)['Pituitary']
            gene_id = pd.read_csv(partition_path)["last_gene_id"]

            # Create a DataFrame with predictions and ground truth

            df = pd.DataFrame({"Prediction": predictions.flatten(), "Ground Truth": ground_truth, "last_gene_id" : gene_id})

            # os.makedirs("results", exist_ok=True)

            # Save the DataFrame to a CSV file
            df.to_csv(os.path.join(results_path, f"ground_truth_{i+1}.csv"), index=False)
        import glob


        # Get a list of all the ground_truth CSV files
        csv_files = glob.glob(os.path.join(results_path,"ground_truth_*.csv"))
        partitions = glob.glob(os.path.join(results_path, "partition_*.csv"))
        # Initialize an empty list to hold DataFrames
        dfs = []

        # Loop through the CSV files and read them into DataFrames
        for file in csv_files:
            df = pd.read_csv(file)
            dfs.append(df)

        # Concatenate all DataFrames, ignoring the index to avoid duplicate index issues
        merged_df = pd.concat(dfs, ignore_index=True)

        # Write the concatenated DataFrame to a new CSV file, ensuring only one header is written
        merged_df.to_csv(os.path.join(results_path, "predictions.csv"), index=False)

        # Delete the original files
        for file in csv_files:
            os.remove(file)
        for file in partitions:
            os.remove(file)
        print("Merging complete and original files deleted.")

        import matplotlib.pyplot as plt
        from sklearn.metrics import mean_squared_error
        from scipy.stats import pearsonr

        
        # Extract the prediction and ground truth columns
        predictions = merged_df['Prediction']
        ground_truth = merged_df['Ground Truth']

        # Calculate MSE
        mse = mean_squared_error(ground_truth, predictions)
        print(f'Mean Squared Error: {mse}')

        # Calculate PCC
        pcc, _ = pearsonr(ground_truth, predictions)
        print(f'Pearson Correlation Coefficient: {pcc}')

        # Save metrics to a file
        with open(os.path.join(results_path,'metrics.txt'), 'w') as f:
            f.write(f'Mean Squared Error: {mse}\n')
            f.write(f'Pearson Correlation Coefficient: {pcc}\n')

        # Plot predictions vs ground truth
        plt.figure(figsize=(8, 6))
        plt.scatter(ground_truth, predictions, alpha=0.5)
        plt.plot([ground_truth.min(), ground_truth.max()], [ground_truth.min(), ground_truth.max()], 'r--', lw=2)
        plt.xlabel('Ground Truth')
        plt.ylabel('Predictions')
        plt.title('Predictions vs Ground Truth')
        plt.savefig(os.path.join(results_path, 'predictions_vs_ground_truth_plot.png'))
        plt.show()



        


if __name__ == "__main__":
    train()
