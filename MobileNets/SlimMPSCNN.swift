//
//  SlimMPSCNN.swift
//  MobileNets
//
//  Created by Wenbo Huang on 17/6/2.
//  Copyright © 2017年 Hollance. All rights reserved.
//

/*
	Copyright (C) 2016 Apple Inc. All Rights Reserved.
	See LICENSE.txt for this sample’s licensing information
	
	Abstract:
	This file describes slimmer routines to create some common MPSCNNFunctions, it is useful especially to fetch network parameters from .dat files
 */

import Foundation
import MetalPerformanceShaders

/**
 This depends on MetalPerformanceShaders.framework
 
 The SlimMPSCNNConvolution is a wrapper class around MPSCNNConvolution used to encapsulate:
 - making an MPSCNNConvolutionDescriptor,
 - adding network parameters (weights and bias binaries by memory mapping the binaries)
 - getting our convolution layer
 */
class SlimMPSCNNConvolution: MPSCNNConvolution{
    
    /**
     A property to keep info from init time whether we will pad input image or not for use during encode call
     */
    private var padding = true
    
    /**
     Initializes a fully connected kernel.
     
     - Parameters:
     - kernelWidth: Kernel Width
     - kernelHeight: Kernel Height
     - inputFeatureChannels: Number feature channels in input of this layer
     - outputFeatureChannels: Number feature channels from output of this layer
     - neuronFilter: A neuronFilter to add at the end as activation, default is nil
     - device: The MTLDevice on which this SlimMPSCNNConvolution filter will be used
     - kernelParamsBinaryName: name of the layer to fetch kernelParameters by adding a prefix "weights_" or "bias_"
     - padding: Bool value whether to use padding or not
     - strideXY: Stride of the filter
     - destinationFeatureChannelOffset: FeatureChannel no. in the destination MPSImage to start writing from, helps with concat operations
     - groupNum: if grouping is used, default value is 1 meaning no groups
     
     - Returns:
     A valid SlimMPSCNNConvolution object or nil, if failure.
     */
    
    
    init(kernelWidth: UInt, kernelHeight: UInt, inputFeatureChannels: UInt, outputFeatureChannels: UInt, neuronFilter: MPSCNNNeuron? = nil, device: MTLDevice, weights: UnsafePointer<Float>,bias: UnsafePointer<Float>, padding willPad: Bool = true, strideXY: (UInt, UInt) = (1, 1), destinationFeatureChannelOffset: UInt = 0, groupNum: UInt = 1){
        

        // create appropriate convolution descriptor with appropriate stride
        let convDesc = MPSCNNConvolutionDescriptor(kernelWidth: Int(kernelWidth),
                                                   kernelHeight: Int(kernelHeight),
                                                   inputFeatureChannels: Int(inputFeatureChannels),
                                                   outputFeatureChannels: Int(outputFeatureChannels),
                                                   neuronFilter: neuronFilter)
        convDesc.strideInPixelsX = Int(strideXY.0)
        convDesc.strideInPixelsY = Int(strideXY.1)
        
        assert((groupNum > 0), "Group size can't be less than 1")
        convDesc.groups = Int(groupNum)
        
        // initialize the convolution layer by calling the parent's (MPSCNNConvlution's) initializer
        super.init(device: device,
                   convolutionDescriptor: convDesc,
                   kernelWeights: weights,
                   biasTerms: bias,
                   flags: MPSCNNConvolutionFlags.none)
        self.destinationFeatureChannelOffset = Int(destinationFeatureChannelOffset)
        
        
        // set padding for calculation of offset during encode call
        padding = willPad
        //padding = true
        self.edgeMode = .zero
        
        
        
    }
    
    /**
     Encode a MPSCNNKernel into a command Buffer. The operation shall proceed out-of-place.
     
     We calculate the appropriate offset as per how TensorFlow calculates its padding using input image size and stride here.
     
     This [Link](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn.py) has an explanation in header comments how tensorFlow pads its convolution input images.
     
     - Parameters:
     - commandBuffer: A valid MTLCommandBuffer to receive the encoded filter
     - sourceImage: A valid MPSImage object containing the source image.
     - destinationImage: A valid MPSImage to be overwritten by result image. destinationImage may not alias sourceImage
     */
    override func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage) {
        
        // select offset according to padding being used or not
        if(padding){
            let pad_along_height = ((destinationImage.height - 1) * strideInPixelsY + kernelHeight - sourceImage.height)
            let pad_along_width  = ((destinationImage.width - 1) * strideInPixelsX + kernelWidth - sourceImage.width)
            let pad_top = Int(pad_along_height / 2)
            let pad_left = Int(pad_along_width / 2)
            
            self.offset = MPSOffset(x: ((Int(kernelWidth)/2) - pad_left), y: (Int(kernelHeight/2) - pad_top), z: 0)
        }
        else{
            self.offset = MPSOffset(x: Int(kernelWidth)/2, y: Int(kernelHeight)/2, z: 0)
        }
        super.encode(commandBuffer: commandBuffer, sourceImage: sourceImage, destinationImage: destinationImage)
    }
    
}


/**
 This depends on MetalPerformanceShaders.framework
 
 The SlimMPSCNNFullyConnected is a wrapper class around MPSCNNFullyConnected used to encapsulate:
 - making an MPSCNNConvolutionDescriptor,
 - adding network parameters (weights and bias binaries by memory mapping the binaries)
 - getting our fullyConnected layer
 */

class SlimMPSCNNFullyConnected: MPSCNNFullyConnected{
    
    
    
    /**
     Initializes a fully connected kernel.
     
     - Parameters:
     - kernelWidth: Kernel Width
     - kernelHeight: Kernel Height
     - inputFeatureChannels: Number feature channels in input of this layer
     - outputFeatureChannels: Number feature channels from output of this layer
     - neuronFilter: A neuronFilter to add at the end as activation, default is nil
     - device: The MTLDevice on which this SlimMPSCNNConvolution filter will be used
     - kernelParamsBinaryName: name of the layer to fetch kernelParameters by adding a prefix "weights_" or "bias_"
     - destinationFeatureChannelOffset: FeatureChannel no. in the destination MPSImage to start writing from, helps with concat operations
     
     - Returns:
     A valid SlimMPSCNNFullyConnected object or nil, if failure.
     */
    
    init(kernelWidth: UInt, kernelHeight: UInt, inputFeatureChannels: UInt, outputFeatureChannels: UInt, neuronFilter: MPSCNNNeuron? = nil, device: MTLDevice, weights: UnsafePointer<Float>,bias: UnsafePointer<Float>, destinationFeatureChannelOffset: UInt = 0){
        
        
        // create appropriate convolution descriptor (in fully connected, stride is always 1)
        let convDesc = MPSCNNConvolutionDescriptor(kernelWidth: Int(kernelWidth),
                                                   kernelHeight: Int(kernelHeight),
                                                   inputFeatureChannels: Int(inputFeatureChannels),
                                                   outputFeatureChannels: Int(outputFeatureChannels),
                                                   neuronFilter: neuronFilter)
        
        // initialize the convolution layer by calling the parent's (MPSCNNFullyConnected's) initializer
        super.init(device: device,
                   convolutionDescriptor: convDesc,
                   kernelWeights: weights,
                   biasTerms: bias,
                   flags: MPSCNNConvolutionFlags.none)
        self.destinationFeatureChannelOffset = Int(destinationFeatureChannelOffset)
        
    }
    
}
