### MDAN: Multiple Source Domain Adaptation with Adversarial Learning for MNIST

MNIST contains four data source, Mnist, SVHN, MNIST\_M and SythDigits.

#### Data Preperration
Downloading address:
[MNIST](http://yann.lecun.com/exdb/mnist/index.html)
[SVHN](http://ufldl.stanford.edu/housenumbers/)
[MNIST\_M](http://yaroslav.ganin.net/)
[SythDigits](http://yaroslav.ganin.net/)

Please download and decompress thoes datasets and put them in a reletive localtion called

~~~
../../data/mnist_data/
~~~

#### Train
~~~
python train.py --config_file_path=config
~~~