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

    

    

def test():
    
    os.environ["TOKENIZERS_PARALLELISM"] = "true" 

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "zhihan1996/DNABERT-2-117M",
        cache_dir=None,
        model_max_length=1002,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

  
    # define datasets and data collator
    

    test_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path="DATASETS/GRCH38/1000tss1000/test.csv", 
                                     kmer=-1)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)


    # load model
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "finetune/no_zero/checkpoint-2600",
        cache_dir=None,
        num_labels=1,
        trust_remote_code=True,
    )



    # configure LoRA
    trainer = transformers.Trainer(model=model,
                               compute_metrics=compute_metrics,
                               data_collator=data_collator,
                            
                              )
  


    import pandas as pd

    # Load the CSV file
    file_path = "DATASETS/GRCH38/1000tss1000/test.csv"  # Replace with your actual file path
    df = pd.read_csv(file_path)

    # Determine the number of rows in the CSV file
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

    # Save each partition as a separate CSV file
    for i, partition in enumerate(partitions):
        partition.to_csv(f'partition_{i + 1}.csv', index=False)
        test_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=f"partition_{i+1}.csv", 
                                     kmer=-1)

        # get the evaluation results from trainer
        results = trainer.predict(test_dataset)
        predictions = results[0][0].reshape((len(results[0][0]), 1))
        import pandas as pd 
        ground_truth = pd.read_csv(f"partition_{i+1}.csv")['Pituitary']
        gene_id = pd.read_csv(f"partition_{i+1}.csv")["last_gene_id"]

        # Create a DataFrame with predictions and ground truth

        df = pd.DataFrame({"Prediction": predictions.flatten(), "Ground Truth": ground_truth, "last_gene_id" : gene_id})

        # os.makedirs("results", exist_ok=True)

        # Save the DataFrame to a CSV file
        df.to_csv(f"ground_truth_{i+1}.csv", index=False)
    import glob


    # Get a list of all the ground_truth CSV files
    csv_files = glob.glob("ground_truth_*.csv")
    partitions = glob.glob("partition_*.csv")
    # Initialize an empty list to hold DataFrames
    dfs = []

    # Loop through the CSV files and read them into DataFrames
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)

    # Concatenate all DataFrames, ignoring the index to avoid duplicate index issues
    merged_df = pd.concat(dfs, ignore_index=True)

    # Write the concatenated DataFrame to a new CSV file, ensuring only one header is written
    merged_df.to_csv("predictions.csv", index=False)

    # Delete the original files
    for file in csv_files:
        os.remove(file)
    for file in partitions:
        os.remove(file)
    print("Merging complete and original files deleted.")

    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error
    from scipy.stats import pearsonr

    # Load the CSV file into a DataFrame
    merged_df = pd.read_csv('predictions.csv')

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
    with open('metrics.txt', 'w') as f:
        f.write(f'Mean Squared Error: {mse}\n')
        f.write(f'Pearson Correlation Coefficient: {pcc}\n')

    # Plot predictions vs ground truth
    plt.figure(figsize=(8, 6))
    plt.scatter(ground_truth, predictions, alpha=0.5)
    plt.plot([ground_truth.min(), ground_truth.max()], [ground_truth.min(), ground_truth.max()], 'r--', lw=2)
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.title('Predictions vs Ground Truth')
    plt.savefig('predictions_vs_ground_truth_plot.png')
    plt.show()






if __name__ == "__main__":
    test()
