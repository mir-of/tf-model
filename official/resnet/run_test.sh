python imagenet_main.py \
--model_dir=./test_output_dir \
--loss_scale=1 \
--data_format='channels_first' \
--num_gpus=1 \
--batch_size=1 \
--train_epochs=1 \
--max_train_steps=1 \
--data_dir=/home/qiaojing/tmp/dataset/tf_PNG1 \
--pretrained_model_checkpoint_path=/home/qiaojing/tmp/resnet_model/tf_ckpt 


#--data_dir=/home/qiaojing/tmp/dataset/tf_imagenet_224_5pic \
