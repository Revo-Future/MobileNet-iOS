# MobileNet-iOS
Google's MobileNet on iOS. Specifically, mobileNet is implemented by MPSCNN which use the Metal to improve the CNN performence. MobileNet architecture comes from the paper [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.](https://arxiv.org/abs/1704.04861v1)

The mobileNet is trained by Caffe on Server, and then convert the weights to MPSCNN weight format. This part is finished by my colleague.

For ordinary convolution in MPSCNN, weight format is **[outputChannel][kernelHeight][kernelWidth][inputChannel]**
For depthwise convolution, weight format is **[kernelHeight][kernelWidth][featureChannels]**

The trained mobileNet model is 16.9M, which is pretty small. And the inference time on iphone 6s plus is about 50ms, it's pretty fast. For iphone 7, it must be faster.

# Requirements
Xcode 8.0

iOS 10.0+

# Example
![image](https://github.com/Revo-Future/MobileNet-iOS/blob/master/example.PNG)

# Notice
This project can't run on iOS simulator, as iOS simulator don't support metal.

# Reference
* https://developer.apple.com/library/content/samplecode/MPSCNNHelloWorld/Introduction/Intro.html#//apple_ref/doc/uid/TP40017482
* https://developer.apple.com/library/content/samplecode/MetalImageRecognition/Introduction/Intro.html
* https://stackoverflow.com/questions/40522224/mpscnn-weight-ordering
* https://github.com/hollance/Forge
* https://github.com/shicai/MobileNet-Caffe
