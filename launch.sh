# VGG16, FACES, STL, T0. 
python main.py \
--gpu_num 0 \
--epochs 2 \
--batch_size 64 \
--num_workers 4 \
--learning_rate 0.001 \
--feature_extractor vgg \
--dataset_path DATASET_PATH_HERE \
--dataset_name faces \
--task_ids 0 \
--task_out_sizes 3 \
--debug \

# MobileNetV3, MEDIC, MTL, T0 & T1.
python main.py \
--gpu_num 0 \
--epochs 2 \
--batch_size 32 \
--num_workers 4 \
--learning_rate 0.0001 \
--feature_extractor mobilenet \
--dataset_path DATASET_PATH_HERE \
--dataset_name medic \
--task_ids 0 1 \
--task_out_sizes 3 5 \
--debug \
