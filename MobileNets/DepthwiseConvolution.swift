//
//  DepthwiseConvolution.swift
//  MobileNets
//
//  Created by Wenbo Huang on 17/6/20.
//  Copyright © 2017年 Hollance. All rights reserved.
//

/*
	Copyright (C) 2016 Apple Inc. All Rights Reserved.
	See LICENSE.txt for this sample’s licensing information
	
	Abstract:
	This file describes depthwise convolution layer.
 */

import Metal
import MetalPerformanceShaders
import Accelerate


/**
 These values get passed to the compute kernel.
 */
public struct KernelParams {
    // The dimensions of the input image.
    var inputWidth: UInt16 = 0
    var inputHeight: UInt16 = 0
    var inputFeatureChannels: UInt16 = 0
    var inputSlices: UInt16 = 0
    
    // Where to start reading in the input image. From ForgeKernel's offset.
    var inputOffsetX: Int16 = 0
    var inputOffsetY: Int16 = 0
    var inputOffsetZ: Int16 = 0
    
    // The dimensions of the output image, derived from clipRect.size.
    var outputWidth: UInt16 = 0
    var outputHeight: UInt16 = 0
    var outputFeatureChannels: UInt16 = 0
    var outputSlices: UInt16 = 0
    
    // This is ForgeKernel's destinationFeatureChannelOffset divided by 4.
    var destinationSliceOffset: UInt16 = 0
    
    // Where to start writing in the output image, derived from clipRect.origin.
    var outputOffsetX: Int16 = 0
    var outputOffsetY: Int16 = 0
    var outputOffsetZ: Int16 = 0
    
    // Zero (0) or clamp (1).
    var edgeMode: UInt16 = 0
    
    // Additional parameters for MPSCNNNeurons.
    var neuronA: Float = 0
    var neuronB: Float = 0
}

func configureNeuronType(filter: MPSCNNNeuron?,
                         constants: MTLFunctionConstantValues,
                         params: inout KernelParams) {
    var neuronType: UInt16 = 0
    if let filter = filter as? MPSCNNNeuronReLU {
        neuronType = 1
        params.neuronA = filter.a
    } else if let filter = filter as? MPSCNNNeuronLinear {
        neuronType = 2
        params.neuronA = filter.a
        params.neuronB = filter.b
    } else if filter is MPSCNNNeuronSigmoid {
        neuronType = 3
    } else if let filter = filter as? MPSCNNNeuronTanH {
        neuronType = 4
        params.neuronA = filter.a
        params.neuronB = filter.b
    } else if filter is MPSCNNNeuronAbsolute {
        neuronType = 5
    }
    constants.setConstantValue(&neuronType, type: .ushort, withName: "neuronType")
}

/**
 Depth-wise convolution
 
 Applies a different convolution kernel to each input channel. Only a single
 kernel is applied to each input channel and so the number of output channels
 is the same as the number of input channels.
 
 A depth-wise convolution only performs filtering; it doesn't combine channels
 to create new features like a regular convolution does.
 */
public class DepthwiseConvolutionKernel {
    let pipeline: MTLComputePipelineState
    let weightsBuffer: MTLBuffer
    let biasBuffer: MTLBuffer
    
    
    public let device: MTLDevice
    public let neuron: MPSCNNNeuron?
    
    public var offset = MPSOffset(x: 0, y: 0, z: 0)
    public var clipRect = MPSRectNoClip
    public var destinationFeatureChannelOffset = 0
    public var edgeMode = MPSImageEdgeMode.zero
    var params = KernelParams()
    /**
     Creates a new DepthwiseConvolution object.
     
     - Parameters:
     - channelMultiplier: If this is M, then each input channel has M kernels
     applied to it, resulting in M output channels for each input channel.
     Default is 1.
     - relu: If true, applies a ReLU to the output. Default is false.
     - kernelWeights: The weights should be arranged in memory like this:
     `[kernelHeight][kernelWidth][featureChannels]`.
     - biasTerms: One bias term per channel (optional).
     */
    public init(device: MTLDevice,
                kernelWidth: Int,
                kernelHeight: Int,
                featureChannels: Int,
                strideInPixelsX: Int = 1,
                strideInPixelsY: Int = 1,
                channelMultiplier: Int = 1,
                neuronFilter: MPSCNNNeuron?,
                kernelWeights: UnsafePointer<Float>,
                biasTerms: UnsafePointer<Float>?) {
        
        precondition(kernelWidth == 3 && kernelHeight == 3, "Only 3x3 kernels are currently supported")
        precondition(channelMultiplier == 1, "Channel multipliers are not supported yet")
        
        let inputSlices = (featureChannels + 3) / 4
        let paddedInputChannels = inputSlices * 4
        let count = kernelHeight * kernelWidth * paddedInputChannels
        weightsBuffer = device.makeBuffer(length: MemoryLayout<UInt16>.stride * count)
        
        copy(weights: kernelWeights, to: weightsBuffer, channelFormat: .float16,
             kernelWidth: kernelWidth, kernelHeight: kernelHeight,
             inputFeatureChannels: featureChannels, outputFeatureChannels: 1)
        
        biasBuffer = makeBuffer(device: device,
                                channelFormat: .float16,
                                outputFeatureChannels: featureChannels,
                                biasTerms: biasTerms)
        
        var params = KernelParams()
        let constants = MTLFunctionConstantValues()
        configureNeuronType(filter: neuronFilter, constants: constants, params: &params)
        
        var stride = [ UInt16(strideInPixelsX), UInt16(strideInPixelsY) ]
        constants.setConstantValue(&stride, type: .ushort2, withName: "stride")
        
        let functionName: String
        if featureChannels <= 4 {
            functionName = "depthwiseConv3x3"
        } else {
            functionName = "depthwiseConv3x3_array"
        }
        pipeline = makeFunction(device: device, name: functionName,
                                constantValues: constants, useForgeLibrary: false)
        self.device = device
        self.neuron = neuronFilter
        self.params = params
        //super.init(device: device, neuron: neuronFilter, params: params)
    }
    
    public func encode(commandBuffer: MTLCommandBuffer,
                                sourceImage: MPSImage, destinationImage: MPSImage) {
        // TODO: set the KernelParams based on clipRect, destinationFeatureChannelOffset, edgeMode
        params.inputOffsetX = Int16(offset.x);
        params.inputOffsetY = Int16(offset.y);
        params.inputOffsetZ = Int16(offset.z);
        
        let encoder = commandBuffer.makeComputeCommandEncoder()
        encoder.setComputePipelineState(pipeline)
        encoder.setTexture(sourceImage.texture, at: 0)
        encoder.setTexture(destinationImage.texture, at: 1)
        encoder.setBytes(&params, length: MemoryLayout<KernelParams>.size, at: 0)
        encoder.setBuffer(weightsBuffer, offset: 0, at: 1)
        encoder.setBuffer(biasBuffer, offset: 0, at: 2)
        encoder.dispatch(pipeline: pipeline, image: destinationImage)
        encoder.endEncoding()
        
        if let image = sourceImage as? MPSTemporaryImage {
            image.readCount -= 1
        }
    }
}
