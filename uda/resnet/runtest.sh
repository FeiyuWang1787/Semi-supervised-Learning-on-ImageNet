python3 ./adanet-resnet.py \
--data /home/ameron/projects/imagenet/datasets/ilsvrc12 \
-d 50  --mode resnet --batch 192 --gpu 3 --eval 1 \
--load './train_log/imagenet-resnet-d50-batch192/model-399960.data-00000-of-00001'

