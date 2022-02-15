# vgg-ssd-tensorRT
Implement VGG-SSD tensor layer by using TensorRT API 
* This project is based on [qfgaohao/pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd) and [tjuskyzhang/mobilenetv1-ssd-tensorrt](https://github.com/tjuskyzhang/mobilenetv1-ssd-tensorrt). This project tested on jetson agx xavier using TensorRt 8.0.1.6 Cuda10.2 and costs about 1240ms on the contrasts of 2.8s by using python.

Due to time, I did not complete the training of the model,you can train your own model by your datasets and generates wts weights using pthTowts.py
## Excute
build and run
```
mkdir build
cd build
cmake ..
make
```


// Serialize the model and generate ssd_mobilenet.engine
```
  ./vgg-ssd-tensorrt -s
```

// Deserialize and generate the detection results test.jpg and so on.


```
  ./vgg-tensorrt -d ../samples
```
