python3 ./adanet-resnet.py \
--data /home/ameron/projects/imagenet/datasets/ilsvrc12 \
-d 18  --mode preact --batch 256 --unlabeled_batch_mult 1 \
--weight 0.5 --gpu 0,1,2,3
