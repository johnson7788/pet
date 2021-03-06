 python3 cli.py \
--method pet \
--pattern_ids 0 1 2 3 4 5 \
--data_dir datasets/agnews-data \
--model_type roberta \
--model_name_or_path roberta-large \
--task_name agnews \
--output_dir outputs/agnews-output \
--do_train \
--do_eval \
--train_examples 10 \
--unlabeled_examples 40000 \
--split_examples_evenly \
--pet_per_gpu_train_batch_size 1 \
--pet_per_gpu_unlabeled_batch_size 3 \
--pet_gradient_accumulation_steps 4 \
--pet_max_steps 250 \
--lm_training \
--sc_per_gpu_train_batch_size 4 \
--sc_per_gpu_unlabeled_batch_size 4 \
--sc_gradient_accumulation_steps 4 \
--sc_max_steps 5000