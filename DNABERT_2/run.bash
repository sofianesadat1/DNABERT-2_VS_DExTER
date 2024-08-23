# cd finetune  

 

# export LR=3e-5  
  
# # Define the base data path and output directory  
# base_data_path="../../DATASETS/GRCH38"  
# base_output_dir="Arnaud_experiments"  
  
# # Array of dataset directories (replace these with your actual dataset directories)  
# datasets=("tss_2000" "2000_tss" "tss_8750" "8750_tss")  
  
# # Loop over each dataset  
# # Loop over each dataset  
# for dataset in "${datasets[@]}"; do  
#   # Extract numbers from dataset name and calculate MAX_LENGTH  
#   SUM=$(echo ${dataset} | grep -o -E '[0-9]+' | awk '{s+=$1} END {print s}')  
#   export MAX_LENGTH=$(($SUM / 4))  
  
#   export DATA_PATH="${base_data_path}/${dataset}"  
#   export OUTPUT_DIR="${base_output_dir}/${dataset}_output"  
  
#   # Training use DataParallel  
#   python train.py \
#     --model_name_or_path zhihan1996/DNABERT-2-117M \
#     --data_path  ${DATA_PATH} \
#     --kmer -1 \
#     --run_name last \
#     --model_max_length ${MAX_LENGTH} \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 16 \
#     --gradient_accumulation_steps 1 \
#     --eval_accumulation_steps 1 \
#     --learning_rate ${LR} \
#     --fp16 \
#     --num_train_epochs 10 \
#     --save_steps 200 \
#     --output_dir ${OUTPUT_DIR} \
#     --evaluation_strategy steps \
#     --eval_steps 200 \
#     --warmup_steps 50 \
#     --logging_steps 100 \
#     --overwrite_output_dir True \
#     --log_level info \
#     --find_unused_parameters False
# done  
  
cd finetune

export LR=3e-5

# Define the base data path and output directory
base_data_path="../../DATASETS/GRCH38"
base_output_dir="GRCH38"

# Array of dataset directories (replace these with your actual dataset directories)
datasets=("2000_tss_2000" "2000_tss" "tss_8750" "8750_tss")

# Array of max lengths corresponding to each dataset
max_lengths=(1000 1024 2048 4096)  # Adjust these values as needed 0.25 of the max length of your sequences 

# Loop over each dataset with its corresponding max length
for i in "${!datasets[@]}"; do
  dataset="${datasets[$i]}"
  max_length="${max_lengths[$i]}"

  export MAX_LENGTH=$max_length
  export DATA_PATH="${base_data_path}/${dataset}/dnabert_2"
  export OUTPUT_DIR="${base_output_dir}/${dataset}_output"

  # Training use DataParallel
  python train.py \
    --model_name_or_path zhihan1996/DNABERT-2-117M \
    --data_path  ${DATA_PATH} \
    --kmer -1 \
    --run_name last \
    --model_max_length ${MAX_LENGTH} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 1 \
    --learning_rate ${LR} \
    --fp16 \
    --num_train_epochs 10 \
    --save_steps 200 \
    --output_dir ${OUTPUT_DIR} \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --warmup_steps 50 \
    --logging_steps 100 \
    --overwrite_output_dir True \
    --log_level info \
    --find_unused_parameters False
done