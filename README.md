# Distributed-tensorflow-Learning-using-CNN-CIFAR-10-dataset-

## This program runs distributed training on CIFAR-10 dataset to classify images using a Convolutional Neural Network. 

## How does it work? 
* You have to define parameter servers and workers. Also assign task to workers to unearth a distributed training system. 

## How to do so? 
An example:
```
Terminal1# python <FILENAME> --ps_hosts=localhost:2222 --worker_hosts=localhost:2223,localhost:2224 --job_name=ps --task_index=0
Terminal2# python <FILENAME> --ps_hosts=localhost:2222 --worker_hosts=localhost:2223,localhost:2224 --job_name=worker --task_index=0
Terminal3# python <FILENAME> --ps_hosts=localhost:2222 --worker_hosts=localhost:2223,localhost:2224 --job_name=worker --task_index=1
```
