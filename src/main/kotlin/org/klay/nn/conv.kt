package org.klay.nn

import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer

fun NeuralNetConfiguration.ListBuilder.conv2d(init: ConvolutionLayer.Builder.() -> Unit) {
    this.layer(ConvolutionLayer.Builder().apply(init).build())
}

fun NeuralNetConfiguration.ListBuilder.subsampling(init: SubsamplingLayer.Builder.() -> Unit) {
    this.layer(SubsamplingLayer.Builder().apply(init).build())
}