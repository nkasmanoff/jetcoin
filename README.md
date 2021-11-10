# jetcoin

Crypto trader using NVIDIA Jetson to train on and track prices.

Steps to run this:

First, build the docker container with the commands below.

```bash
$ sudo docker build -t nsk367/jetcoin .
```


```bash
$ sudo docker run --runtime nvidia -it --rm --network host -v ~/jetcoin:/home/noah/jetcoin nsk367/jetcoin
```


Next, let's train a model. It is worth noting this is not really a serious project on price prediction, so let's leave the hparams and run with little attention to that. After cd'ing into the src directory, run the following commands.


```bash
$ python train.py --gpus 1
```


Now that we have some model trained, the continuous prediction begins


```bash
$ python continous_prediction.py
```


After specifying the model path, this will continuously predict and download prices to see how well the model actually does after looking at the produced csv files.


On the wish list is how to automatically grab that model name, along with other automation steps briefly discussed below.




More things to do include scaling to difference coins, looking at different resolutions,
and adding more tracking capabilities. Intent for this is to see how efficiently
I can take a project end to end on Jetson.
