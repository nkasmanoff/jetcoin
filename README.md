# jetcoin

Crypto trader using NVIDIA Jetson to train and track prices.


sudo docker build -t nsk367/jetcoin .



sudo docker run --runtime nvidia -it --rm --network host --volume ~/jetcoin:/jetcoin nsk367/jetcoin


and once inside can train, test, and predict.


More things to do include scaling to difference coins, looking at different resolutions,
and adding more tracking capabilities. Intent for this is to see how efficiently
I can take a project end to end on Jetson.
