import MetalPerformanceShaders
import QuartzCore

private func makePool(device: MTLDevice) -> MPSCNNPoolingAverage {
    // only one pooling layer in MobileNet, are max pool, 7x7, stride 7.
    
    let pool = MPSCNNPoolingAverage(device: device,
                                    kernelWidth: 7,
                                    kernelHeight: 7,
                                    strideInPixelsX: 7,
                                    strideInPixelsY: 7)
    pool.offset = MPSOffset(x: 3, y: 3, z: 0)
    //pool.edgeMode = MPSImageEdgeMode.clamp
    return pool
}



/*
 Implements the MobileNet.
 The neural network from the paper "MobileNets: Efficient Convolutional Neural
 Networks for Mobile Vision Applications" https://arxiv.org/abs/1704.04861v1
 */
public class MobileNet {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    
    // The custom compute kernels for preprocessing the input images.
    let pipelineRGB: MTLComputePipelineState
    let pipelineBGR: MTLComputePipelineState
    
    let outputImage: MPSImage
    
    // The neural network expects a 224x224 pixel image. We use a lanczos filter
    // to scale the input image down to these dimensions.
    let lanczos: MPSImageLanczosScale
    
    // After the last layer (fc7), we take the "softmax" of each output neuron.
    // This converts the last layer into a 1000-element vector of probabilities,
    // where each element in this vector corresponds to an ImageNet class label.
    let softmax: MPSCNNSoftMax
    
    /* The layers in the network: */
    
    
    let conv1_s2: MPSCNNConvolution  // 224x224x3  input, kernels (3x3x3x32  = 864 weights + 32 bias). s=2,p=1
    
    let conv2_1_dw: DepthwiseConvolutionKernel  // 112x112x32 input, kernels (3x3x32 = 288 weights + 32 bias) s=1,p=1
    let conv2_1_s1: MPSCNNConvolution  // 112x112x32 input, kernels (1x1x32x64 = 2048 weights + 64 bias) s=1,p=0
    let conv2_2_dw: DepthwiseConvolutionKernel // 112x112x64 input, kernels (3x3x64 = 576 weights + 64 bias) s=2,p=1
    let conv2_2_s1: MPSCNNConvolution // 56x56x64 input, kernels (1x1x64x128 = 8912 weights + 128 bias) s=1,p=0
    
    let conv3_1_dw: DepthwiseConvolutionKernel // 56x56x128 input, kernels (3x3x128 = 1152 weights + 128 bias) s=1,p=1
    let conv3_1_s1: MPSCNNConvolution // 56x56x128 input, kernels (1x1x128x128 = 16384 weights + 128 bias) s=1,p=0
    let conv3_2_dw: DepthwiseConvolutionKernel // 56x56x128 input, kernels (3x3x128 = 1152 weights + 128 bias) s=2,p=1
    let conv3_2_s1: MPSCNNConvolution // 28x28x128 input, kernels (1x1x128x256 = 32768 weights + 256 bias) s=1,p=0
    
    let conv4_1_dw: DepthwiseConvolutionKernel // 28x28x256 input, kernels (3x3x256 = 2304 weights + 256 bias) s=1,p=1
    let conv4_1_s1: MPSCNNConvolution // 28x28x256 input, kernels (1x1x256x256 = 65536 weights + 256 bias) s=1,p=0
    let conv4_2_dw: DepthwiseConvolutionKernel // 28x28x256 input, kernels (3x3x256 = 2304 weights + 256 bias) s=2,p=1
    let conv4_2_s1: MPSCNNConvolution // 14x14x256 input, kernels (1x1x256x512 = 131072 weights + 512 bias) s=1,p=0
    
    let conv5_1_dw: DepthwiseConvolutionKernel // 14x14x512 input, kernels (3x3x512 = 4608 weights + 512 bias) s=1,p=1
    let conv5_1_s1: MPSCNNConvolution // 14x14x512 input, kernels (1x1x512x512 = 262144 weights + 512 bias) s=1,p=0
    let conv5_2_dw: DepthwiseConvolutionKernel // 14x14x512 input, kernels (3x3x512 = 4608 weights + 512 bias) s=1,p=1
    let conv5_2_s1: MPSCNNConvolution // 14x14x512 input, kernels (1x1x512x512 = 262144 weights + 512 bias) s=1,p=0
    let conv5_3_dw: DepthwiseConvolutionKernel // 14x14x512 input, kernels (3x3x512 = 4608 weights + 512 bias) s=1,p=1
    let conv5_3_s1: MPSCNNConvolution // 14x14x512 input, kernels (1x1x512x512 = 262144 weights + 512 bias) s=1,p=0
    let conv5_4_dw: DepthwiseConvolutionKernel // 14x14x512 input, kernels (3x3x512 = 4608 weights + 512 bias) s=1,p=1
    let conv5_4_s1: MPSCNNConvolution // 14x14x512 input, kernels (1x1x512x512 = 262144 weights + 512 bias) s=1,p=0
    let conv5_5_dw: DepthwiseConvolutionKernel // 14x14x512 input, kernels (3x3x512 = 4608 weights + 512 bias) s=1,p=1
    let conv5_5_s1: MPSCNNConvolution // 14x14x512 input, kernels (1x1x512x512 = 262144 weights + 512 bias) s=1,p=0
    let conv5_6_dw: DepthwiseConvolutionKernel // 14x14x512 input, kernels (3x3x512 = 4608 weights + 512 bias) s=2,p=1
    let conv5_6_s1: MPSCNNConvolution // 7x7x512 input, kernels (1x1x512x1024 = 524288 weights + 1024 bias) s=1,p=0
    
    let conv6_1_dw: DepthwiseConvolutionKernel // 7x7x1024 input, kernels (3x3x1024 = 9216 weights + 1024 bias) s=1,p=1
    let conv6_1_s1: MPSCNNConvolution // 7x7x1024 input, kernels (1x1x1024x1024 = 1048576 weights + 1024 bias) s=1,p=0
    let pool6: MPSCNNPoolingAverage   // 7x7x1024 input ->1x1x1024 output, caffe global_pooling: true
    let fc7: MPSCNNConvolution   //  fc weights (1x1x1024x1000 = 1024000 weights + 1000 bias)
    
    
    
    /* These MPSImage descriptors tell the network about the sizes of the data
     volumes that flow between the layers. */
    
    let input_id  = MPSImageDescriptor(channelFormat: .float16, width: 224, height: 224, featureChannels: 3)
    let conv1_id  = MPSImageDescriptor(channelFormat: .float16, width: 112, height: 112, featureChannels: 32)
    let conv2_1dw_id = MPSImageDescriptor(channelFormat: .float16, width: 112, height: 112, featureChannels: 32)
    let conv2_1s_id  = MPSImageDescriptor(channelFormat: .float16, width: 112, height: 112, featureChannels: 64)
    let conv2_2dw_id = MPSImageDescriptor(channelFormat: .float16, width: 56, height: 56, featureChannels: 64)
    let conv2_2s_id =  MPSImageDescriptor(channelFormat: .float16, width: 56, height: 56, featureChannels: 128)
    
    let conv3_1dw_id = MPSImageDescriptor(channelFormat: .float16, width: 56, height: 56, featureChannels: 128)
    let conv3_1s_id =  MPSImageDescriptor(channelFormat: .float16, width: 56, height: 56, featureChannels: 128)
    let conv3_2dw_id = MPSImageDescriptor(channelFormat: .float16, width: 28, height: 28, featureChannels: 128)
    let conv3_2s_id =  MPSImageDescriptor(channelFormat: .float16, width: 28, height: 28, featureChannels: 256)
    
    let conv4_1dw_id = MPSImageDescriptor(channelFormat: .float16, width: 28, height: 28, featureChannels: 256)
    let conv4_1s_id  = MPSImageDescriptor(channelFormat: .float16, width: 28, height: 28, featureChannels: 256)
    let conv4_2dw_id = MPSImageDescriptor(channelFormat: .float16, width: 14, height: 14, featureChannels: 256)
    let conv4_2s_id  = MPSImageDescriptor(channelFormat: .float16, width: 14, height: 14, featureChannels: 512)
    
    let conv5_dw_id  = MPSImageDescriptor(channelFormat: .float16, width: 14, height: 14, featureChannels: 512)
    let conv5_s_id   = MPSImageDescriptor(channelFormat: .float16, width: 14, height: 14, featureChannels: 512)
    let conv5_6dw_id = MPSImageDescriptor(channelFormat: .float16, width: 7, height: 7, featureChannels: 512)
    let conv5_6s_id  = MPSImageDescriptor(channelFormat: .float16, width: 7, height: 7, featureChannels: 1024)
    
    let conv6_dw_id  = MPSImageDescriptor(channelFormat: .float16, width: 7, height: 7, featureChannels: 1024)
    let conv6_s_id   = MPSImageDescriptor(channelFormat: .float16, width: 7, height: 7, featureChannels: 1024)
    
    let pool6_id     = MPSImageDescriptor(channelFormat: .float16, width: 1, height: 1, featureChannels: 1024)
    let output_id = MPSImageDescriptor(channelFormat: .float16, width:   1, height:   1, featureChannels: 1000)
    
    
    
    let labels = MobileNetsLabels()
    
    public init(device: MTLDevice) {
        print("Setting up neural network...")
        let startTime = CACurrentMediaTime()
        
        self.device = device
        commandQueue = device.makeCommandQueue()
        
        outputImage = MPSImage(device: device, imageDescriptor: output_id)
        
        // Before we pass an image into the network, we need to adjust its RGB
        // values. This is done with a custom compute kernel. Here we load that
        // kernel (from Shaders.metal) and set up the compute pipeline.
        do {
            let library = device.newDefaultLibrary()!
            let adjust_mean_rgb = library.makeFunction(name: "adjust_mean_rgb")
            pipelineRGB = try device.makeComputePipelineState(function: adjust_mean_rgb!)
            
            let adjust_mean_bgr = library.makeFunction(name: "adjust_mean_bgr")
            pipelineBGR = try device.makeComputePipelineState(function: adjust_mean_bgr!)
        } catch {
            fatalError("Error initializing compute pipeline")
        }
        
        // Uncomment this to test the network with all zero weights.
        //let blob = MobileNetsData()
        guard let path = Bundle.main.path(forResource: "MobileNet_weights", ofType: "bat"),
            let blob = MobileNetsData(path: path) else {
                fatalError("Error loading network parameters")
        }
        
        lanczos = MPSImageLanczosScale(device: device)
        
        let relu = MPSCNNNeuronReLU(device: device, a: 0)
        conv1_s2 = SlimMPSCNNConvolution(kernelWidth: 3,
                                         kernelHeight: 3,
                                         inputFeatureChannels: 3,
                                         outputFeatureChannels: 32,
                                         neuronFilter: relu,
                                         device: device,
                                         weights:blob.conv1_s2_w,
                                         bias: blob.conv1_s2_b,
                                         padding: true,
                                         strideXY: (2,2)
                                         )
        conv2_1_dw = DepthwiseConvolutionKernel(device: device,
                                                kernelWidth: 3,
                                                kernelHeight: 3,
                                                featureChannels: 32,
                                                strideInPixelsX: 1,
                                                strideInPixelsY: 1,
                                                channelMultiplier: 1,
                                                neuronFilter: relu,
                                                kernelWeights: blob.conv2_1_dw_w,
                                                biasTerms: blob.conv2_1_dw_b)
        
        conv2_1_s1 = SlimMPSCNNConvolution(kernelWidth: 1,
                                           kernelHeight: 1,
                                           inputFeatureChannels: 32,
                                           outputFeatureChannels: 64,
                                           neuronFilter: relu,
                                           device: device,
                                           weights:blob.conv2_1_s1_w,
                                           bias: blob.conv2_1_s1_b,
                                           padding: false,
                                           strideXY: (1,1)
                                           )
        conv2_2_dw = DepthwiseConvolutionKernel(device: device,
                                                kernelWidth: 3,
                                                kernelHeight: 3,
                                                featureChannels: 64,
                                                strideInPixelsX: 2,
                                                strideInPixelsY: 2,
                                                channelMultiplier: 1,
                                                neuronFilter: relu,
                                                kernelWeights: blob.conv2_2_dw_w,
                                                biasTerms: blob.conv2_2_dw_b)
        conv2_2_s1 = SlimMPSCNNConvolution(kernelWidth: 1,
                                           kernelHeight: 1,
                                           inputFeatureChannels: 64,
                                           outputFeatureChannels: 128,
                                           neuronFilter: relu,
                                           device: device,
                                           weights:blob.conv2_2_s1_w,
                                           bias: blob.conv2_2_s1_b,
                                           padding: false,
                                           strideXY: (1,1)
        )
        
        
        conv3_1_dw = DepthwiseConvolutionKernel(device: device,
                                                kernelWidth: 3,
                                                kernelHeight: 3,
                                                featureChannels: 128,
                                                strideInPixelsX: 1,
                                                strideInPixelsY: 1,
                                                channelMultiplier: 1,
                                                neuronFilter: relu,
                                                kernelWeights: blob.conv3_1_dw_w,
                                                biasTerms: blob.conv3_1_dw_b)
        conv3_1_s1 = SlimMPSCNNConvolution(kernelWidth: 1,
                                           kernelHeight: 1,
                                           inputFeatureChannels: 128,
                                           outputFeatureChannels: 128,
                                           neuronFilter: relu,
                                           device: device,
                                           weights:blob.conv3_1_s1_w,
                                           bias: blob.conv3_1_s1_b,
                                           padding: false,
                                           strideXY: (1,1)
        )
        conv3_2_dw = DepthwiseConvolutionKernel(device: device,
                                                kernelWidth: 3,
                                                kernelHeight: 3,
                                                featureChannels: 128,
                                                strideInPixelsX: 2,
                                                strideInPixelsY: 2,
                                                channelMultiplier: 1,
                                                neuronFilter: relu,
                                                kernelWeights: blob.conv3_2_dw_w,
                                                biasTerms: blob.conv3_2_dw_b)
        conv3_2_s1 = SlimMPSCNNConvolution(kernelWidth: 1,
                                          kernelHeight: 1,
                                          inputFeatureChannels: 128,
                                          outputFeatureChannels: 256,
                                          neuronFilter: relu,
                                          device: device,
                                          weights:blob.conv3_2_s1_w,
                                          bias: blob.conv3_2_s1_b,
                                          padding: false,
                                          strideXY: (1,1)
        )
        
        conv4_1_dw = DepthwiseConvolutionKernel(device: device,
                                                kernelWidth: 3,
                                                kernelHeight: 3,
                                                featureChannels: 256,
                                                strideInPixelsX: 1,
                                                strideInPixelsY: 1,
                                                channelMultiplier: 1,
                                                neuronFilter: relu,
                                                kernelWeights: blob.conv4_1_dw_w,
                                                biasTerms: blob.conv4_1_dw_b)
        conv4_1_s1 = SlimMPSCNNConvolution(kernelWidth: 1,
                                           kernelHeight: 1,
                                           inputFeatureChannels: 256,
                                           outputFeatureChannels: 256,
                                           neuronFilter: relu,
                                           device: device,
                                           weights:blob.conv4_1_s1_w,
                                           bias: blob.conv4_1_s1_b,
                                           padding: false,
                                           strideXY: (1,1)
        )
        conv4_2_dw = DepthwiseConvolutionKernel(device: device,
                                                kernelWidth: 3,
                                                kernelHeight: 3,
                                                featureChannels: 256,
                                                strideInPixelsX: 2,
                                                strideInPixelsY: 2,
                                                channelMultiplier: 1,
                                                neuronFilter: relu,
                                                kernelWeights: blob.conv4_2_dw_w,
                                                biasTerms: blob.conv4_2_dw_b)
        conv4_2_s1 = SlimMPSCNNConvolution(kernelWidth: 1,
                                           kernelHeight: 1,
                                           inputFeatureChannels: 256,
                                           outputFeatureChannels: 512,
                                           neuronFilter: relu,
                                           device: device,
                                           weights:blob.conv4_2_s1_w,
                                           bias: blob.conv4_2_s1_b,
                                           padding: false,
                                           strideXY: (1,1)
        )

        conv5_1_dw = DepthwiseConvolutionKernel(device: device,
                                                kernelWidth: 3,
                                                kernelHeight: 3,
                                                featureChannels: 512,
                                                strideInPixelsX: 1,
                                                strideInPixelsY: 1,
                                                channelMultiplier: 1,
                                                neuronFilter: relu,
                                                kernelWeights: blob.conv5_1_dw_w,
                                                biasTerms: blob.conv5_1_dw_b)
        conv5_1_s1 = SlimMPSCNNConvolution(kernelWidth: 1,
                                           kernelHeight: 1,
                                           inputFeatureChannels: 512,
                                           outputFeatureChannels: 512,
                                           neuronFilter: relu,
                                           device: device,
                                           weights:blob.conv5_1_s1_w,
                                           bias: blob.conv5_1_s1_b,
                                           padding: false,
                                           strideXY: (1,1)
        )
        
        conv5_2_dw = DepthwiseConvolutionKernel(device: device,
                                                kernelWidth: 3,
                                                kernelHeight: 3,
                                                featureChannels: 512,
                                                strideInPixelsX: 1,
                                                strideInPixelsY: 1,
                                                channelMultiplier: 1,
                                                neuronFilter: relu,
                                                kernelWeights: blob.conv5_2_dw_w,
                                                biasTerms: blob.conv5_2_dw_b)
        conv5_2_s1 = SlimMPSCNNConvolution(kernelWidth: 1,
                                           kernelHeight: 1,
                                           inputFeatureChannels: 512,
                                           outputFeatureChannels: 512,
                                           neuronFilter: relu,
                                           device: device,
                                           weights:blob.conv5_2_s1_w,
                                           bias: blob.conv5_2_s1_b,
                                           padding: false,
                                           strideXY: (1,1)
        )
        
        conv5_3_dw = DepthwiseConvolutionKernel(device: device,
                                                kernelWidth: 3,
                                                kernelHeight: 3,
                                                featureChannels: 512,
                                                strideInPixelsX: 1,
                                                strideInPixelsY: 1,
                                                channelMultiplier: 1,
                                                neuronFilter: relu,
                                                kernelWeights: blob.conv5_3_dw_w,
                                                biasTerms: blob.conv5_3_dw_b)
        conv5_3_s1 = SlimMPSCNNConvolution(kernelWidth: 1,
                                           kernelHeight: 1,
                                           inputFeatureChannels: 512,
                                           outputFeatureChannels: 512,
                                           neuronFilter: relu,
                                           device: device,
                                           weights:blob.conv5_3_s1_w,
                                           bias: blob.conv5_3_s1_b,
                                           padding: false,
                                           strideXY: (1,1)
        )
        conv5_4_dw = DepthwiseConvolutionKernel(device: device,
                                                kernelWidth: 3,
                                                kernelHeight: 3,
                                                featureChannels: 512,
                                                strideInPixelsX: 1,
                                                strideInPixelsY: 1,
                                                channelMultiplier: 1,
                                                neuronFilter: relu,
                                                kernelWeights: blob.conv5_4_dw_w,
                                                biasTerms: blob.conv5_4_dw_b)
        conv5_4_s1 = SlimMPSCNNConvolution(kernelWidth: 1,
                                           kernelHeight: 1,
                                           inputFeatureChannels: 512,
                                           outputFeatureChannels: 512,
                                           neuronFilter: relu,
                                           device: device,
                                           weights:blob.conv5_4_s1_w,
                                           bias: blob.conv5_4_s1_b,
                                           padding: false,
                                           strideXY: (1,1)
        )
        conv5_5_dw = DepthwiseConvolutionKernel(device: device,
                                                kernelWidth: 3,
                                                kernelHeight: 3,
                                                featureChannels: 512,
                                                strideInPixelsX: 1,
                                                strideInPixelsY: 1,
                                                channelMultiplier: 1,
                                                neuronFilter: relu,
                                                kernelWeights: blob.conv5_5_dw_w,
                                                biasTerms: blob.conv5_5_dw_b)
        conv5_5_s1 = SlimMPSCNNConvolution(kernelWidth: 1,
                                           kernelHeight: 1,
                                           inputFeatureChannels: 512,
                                           outputFeatureChannels: 512,
                                           neuronFilter: relu,
                                           device: device,
                                           weights:blob.conv5_5_s1_w,
                                           bias: blob.conv5_5_s1_b,
                                           padding: false,
                                           strideXY: (1,1)
        )
        conv5_6_dw = DepthwiseConvolutionKernel(device: device,
                                                kernelWidth: 3,
                                                kernelHeight: 3,
                                                featureChannels: 512,
                                                strideInPixelsX: 2,
                                                strideInPixelsY: 2,
                                                channelMultiplier: 1,
                                                neuronFilter: relu,
                                                kernelWeights: blob.conv5_6_dw_w,
                                                biasTerms: blob.conv5_6_dw_b)
        conv5_6_s1 = SlimMPSCNNConvolution(kernelWidth: 1,
                                           kernelHeight: 1,
                                           inputFeatureChannels: 512,
                                           outputFeatureChannels: 1024,
                                           neuronFilter: relu,
                                           device: device,
                                           weights:blob.conv5_6_s1_w,
                                           bias: blob.conv5_6_s1_b,
                                           padding: false,
                                           strideXY: (1,1)
        )
   
        conv6_1_dw = DepthwiseConvolutionKernel(device: device,
                                                kernelWidth: 3,
                                                kernelHeight: 3,
                                                featureChannels: 1024,
                                                strideInPixelsX: 1,
                                                strideInPixelsY: 1,
                                                channelMultiplier: 1,
                                                neuronFilter: relu,
                                                kernelWeights: blob.conv6_1_dw_w,
                                                biasTerms: blob.conv6_1_dw_b)
        conv6_1_s1 = SlimMPSCNNConvolution(kernelWidth: 1,
                                           kernelHeight: 1,
                                           inputFeatureChannels: 1024,
                                           outputFeatureChannels: 1024,
                                           neuronFilter: relu,
                                           device: device,
                                           weights:blob.conv6_1_s1_w,
                                           bias: blob.conv6_1_s1_b,
                                           padding: false,
                                           strideXY: (1,1)
        )
        
        pool6 = makePool(device: device)
        //for fc7 layer, conv layer or fc layer is the same.
        fc7 = SlimMPSCNNConvolution(kernelWidth: 1,
                                           kernelHeight: 1,
                                           inputFeatureChannels: 1024,
                                           outputFeatureChannels: 1000,
                                           neuronFilter: nil,
                                           device: device,
                                           weights:blob.fc7_w,
                                           bias: blob.fc7_b,
                                           padding: false,
                                           strideXY: (1,1)
        )
        
//        fc7 = SlimMPSCNNFullyConnected(kernelWidth: 1,
//                                       kernelHeight: 1,
//                                       inputFeatureChannels: 1024,
//                                       outputFeatureChannels: 1000,
//                                       neuronFilter: nil,
//                                       device: device,
//                                       weights:blob.fc7_w,
//                                       bias: blob.fc7_b
//                                       )

        softmax = MPSCNNSoftMax(device: device)
        
        let endTime = CACurrentMediaTime()
        print("Elapsed time: \(endTime - startTime) sec")
    }
    
    /* Performs the inference step. This takes the input image, converts it into
     the format the network expects, then feeds it into the network. The result
     is a 1000-element vector of probabilities. Returns the 5 ImageNet classes
     with the highest predicted probability values. */
    public func predict(image inputImage: MPSImage, bgr: Bool) -> [Prediction] {
        let startTime = CACurrentMediaTime()
        
        autoreleasepool{
            let commandBuffer = commandQueue.makeCommandBuffer()
            
            // This lets us squeeze some extra speed out of Metal.
            MPSTemporaryImage.prefetchStorage(with: commandBuffer, imageDescriptorList: [
                input_id, conv1_id, conv2_1dw_id, conv2_1s_id, conv2_2dw_id, conv2_2s_id, conv3_1dw_id,conv3_1s_id,
                conv3_2dw_id,conv3_2s_id,conv4_1dw_id,conv4_1s_id,conv4_2dw_id,conv4_2s_id,conv5_dw_id,conv5_s_id,
                conv5_6dw_id,conv5_6s_id,conv6_dw_id,conv6_s_id,pool6_id,output_id ])
            
            // Scale the input image to 224x224 pixels.
            let img1 = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: input_id)
            lanczos.encode(commandBuffer: commandBuffer, sourceTexture: inputImage.texture, destinationTexture: img1.texture)
            
            //let img2 = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: input_id)
            let img2 = MPSImage(device: device, imageDescriptor: input_id)
            
            // Adjust the RGB values of each pixel to be in the range -128...127
            // by subtracting the "mean pixel". If the input texture is RGB, this
            // also swaps the R and B values because the model expects BGR pixels.
            // As far as I can tell there is no MPS shader that can do these things,
            // so we use a custom compute kernel.
            let encoder = commandBuffer.makeComputeCommandEncoder()
            encoder.setComputePipelineState(bgr ? pipelineBGR : pipelineRGB)
            encoder.setTexture(img1.texture, at: 0)
            encoder.setTexture(img2.texture, at: 1)
            let threadsPerGroups = MTLSizeMake(8, 8, 1)
            let threadGroups = MTLSizeMake(img2.texture.width / threadsPerGroups.width,
                                           img2.texture.height / threadsPerGroups.height, 1)
            encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadsPerGroups)
            encoder.endEncoding()
            img1.readCount -= 1    // see MPSTemporaryImage docs why this is needed
            
            
            
            // Now we take the output from our custom shader and pass it through the
            // layers of the neural network. For each layer we use a new "temporary"
            // MPSImage to hold the results.
            
            let conv1_s2_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv1_id)
            conv1_s2.encode(commandBuffer: commandBuffer, sourceImage: img2, destinationImage: conv1_s2_img)
            
            let conv2_1dw_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv2_1dw_id)
            conv2_1_dw.encode(commandBuffer: commandBuffer, sourceImage: conv1_s2_img, destinationImage: conv2_1dw_img)
            
            let conv2_1s_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv2_1s_id)
            conv2_1_s1.encode(commandBuffer: commandBuffer, sourceImage: conv2_1dw_img, destinationImage: conv2_1s_img)
            let conv2_2dw_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv2_2dw_id)
            conv2_2_dw.encode(commandBuffer: commandBuffer, sourceImage: conv2_1s_img, destinationImage: conv2_2dw_img)
            let conv2_2s_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv2_2s_id)
            conv2_2_s1.encode(commandBuffer: commandBuffer, sourceImage: conv2_2dw_img, destinationImage: conv2_2s_img)
            
            let conv3_1dw_img  = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv3_1dw_id)
            conv3_1_dw.encode(commandBuffer: commandBuffer, sourceImage: conv2_2s_img, destinationImage: conv3_1dw_img)
            let conv3_1s_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv3_1s_id)
            conv3_1_s1.encode(commandBuffer: commandBuffer, sourceImage: conv3_1dw_img, destinationImage: conv3_1s_img)
            let conv3_2dw_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv3_2dw_id)
            conv3_2_dw.encode(commandBuffer: commandBuffer, sourceImage: conv3_1s_img, destinationImage: conv3_2dw_img)
            let conv3_2s_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv3_2s_id)
            conv3_2_s1.encode(commandBuffer: commandBuffer, sourceImage: conv3_2dw_img, destinationImage: conv3_2s_img)
            
            let conv4_1dw_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv4_1dw_id)
            conv4_1_dw.encode(commandBuffer: commandBuffer, sourceImage: conv3_2s_img, destinationImage: conv4_1dw_img)
            let conv4_1s_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv4_1s_id)
            conv4_1_s1.encode(commandBuffer: commandBuffer, sourceImage: conv4_1dw_img, destinationImage: conv4_1s_img)
            let conv4_2dw_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv4_2dw_id)
            conv4_2_dw.encode(commandBuffer: commandBuffer, sourceImage: conv4_1s_img, destinationImage: conv4_2dw_img)
            let conv4_2s_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv4_2s_id)
            conv4_2_s1.encode(commandBuffer: commandBuffer, sourceImage: conv4_2dw_img, destinationImage: conv4_2s_img)
            
            let conv5_1dw_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_dw_id)
            conv5_1_dw.encode(commandBuffer: commandBuffer, sourceImage: conv4_2s_img, destinationImage: conv5_1dw_img)
            let conv5_1s_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_s_id)
            conv5_1_s1.encode(commandBuffer: commandBuffer, sourceImage: conv5_1dw_img, destinationImage: conv5_1s_img)
            let conv5_2dw_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_dw_id)
            conv5_2_dw.encode(commandBuffer: commandBuffer, sourceImage: conv5_1s_img, destinationImage: conv5_2dw_img)
            let conv5_2s_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_s_id)
            conv5_2_s1.encode(commandBuffer: commandBuffer, sourceImage: conv5_2dw_img, destinationImage: conv5_2s_img)
            let conv5_3dw_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_dw_id)
            conv5_3_dw.encode(commandBuffer: commandBuffer, sourceImage: conv5_2s_img, destinationImage: conv5_3dw_img)
            let conv5_3s_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_s_id)
            conv5_3_s1.encode(commandBuffer: commandBuffer, sourceImage: conv5_3dw_img, destinationImage: conv5_3s_img)
            let conv5_4dw_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_dw_id)
            conv5_4_dw.encode(commandBuffer: commandBuffer, sourceImage: conv5_3s_img, destinationImage: conv5_4dw_img)
            let conv5_4s_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_s_id)
            conv5_4_s1.encode(commandBuffer: commandBuffer, sourceImage: conv5_4dw_img, destinationImage: conv5_4s_img)
            let conv5_5dw_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_dw_id)
            conv5_5_dw.encode(commandBuffer: commandBuffer, sourceImage: conv5_4s_img, destinationImage: conv5_5dw_img)
            let conv5_5s_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_s_id)
            conv5_5_s1.encode(commandBuffer: commandBuffer, sourceImage: conv5_5dw_img, destinationImage: conv5_5s_img)
            let conv5_6dw_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_6dw_id)
            conv5_6_dw.encode(commandBuffer: commandBuffer, sourceImage: conv5_5s_img, destinationImage: conv5_6dw_img)
            let conv5_6s_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_6s_id)
            conv5_6_s1.encode(commandBuffer: commandBuffer, sourceImage: conv5_6dw_img, destinationImage: conv5_6s_img)
            
            let conv6_dw_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv6_dw_id)
            conv6_1_dw.encode(commandBuffer: commandBuffer, sourceImage: conv5_6s_img, destinationImage: conv6_dw_img)
            let conv6_s_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv6_s_id)
            conv6_1_s1.encode(commandBuffer: commandBuffer, sourceImage: conv6_dw_img, destinationImage: conv6_s_img)
            
            let pool6_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool6_id)
            pool6.encode(commandBuffer: commandBuffer, sourceImage: conv6_s_img, destinationImage: pool6_img)
            
            let fc7_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: output_id)
            fc7.encode(commandBuffer: commandBuffer, sourceImage: pool6_img, destinationImage: fc7_img)
            
            // Finally, apply the softmax function to the output of the last layer.
            // The output image is not an MPSTemporaryImage but a regular MSPImage.
            softmax.encode(commandBuffer: commandBuffer, sourceImage: fc7_img, destinationImage: outputImage)
            
            // Tell the GPU to start and wait until it's done.
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        
        }
        
        // Convert the texture from outputImage into something we can use from
        // Swift and then find the ImageNet classes with the highest probability.
        let result = self.labels.top5Labels(prediction: self.outputImage.toFloatArray())
        
        let endTime = CACurrentMediaTime()
        print("Elapsed time: \(endTime - startTime) sec")
        
        return result
    }
}
