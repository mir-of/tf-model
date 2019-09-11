python ../official/resnet/imagenet_main.py \
--model_dir=./test_output_dir \
--loss_scale=1 \
--data_format='channels_first' \
--num_gpus=1 \
--weight_decay=0 \
--batch_size=1 \
--train_epochs=100 \
--pretrained_model_checkpoint_path=/dataset/PNGS/cnns_model_for_test/resnet50/models/tf_model \
--data_dir=/dataset/PNGS/PNG228/tf_record 


#--pretrained_model_checkpoint_path=/home/qiaojing/tmp/resnet_model/tf_ckpt
#--data_dir=/home/qiaojing/tmp/dataset/tf_imagenet_224_5pic \
