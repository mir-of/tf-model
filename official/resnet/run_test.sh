python imagenet_main.py \
--model_dir=./test_output_dir \
--loss_scale=1 \
--data_format='channels_first' \
--num_gpus=1 \
--weight_decay=0 \
--batch_size=1 \
--train_epochs=5 \
--pretrained_model_checkpoint_path=/home/qiaojing/tmp/resnet_model/png_290_model/tf_model \
--data_dir=/home/qiaojing/tmp/dataset/PNG290/tf_record 


#--pretrained_model_checkpoint_path=/home/qiaojing/tmp/resnet_model/tf_ckpt
#--data_dir=/home/qiaojing/tmp/dataset/tf_imagenet_224_5pic \
