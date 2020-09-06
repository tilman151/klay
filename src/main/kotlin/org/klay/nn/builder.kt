package org.klay.nn

import org.deeplearning4j.nn.conf.*
import org.deeplearning4j.nn.conf.layers.*


fun sequential(init: NeuralNetConfiguration.Builder.() -> NeuralNetConfiguration.ListBuilder): MultiLayerConfiguration {
    return NeuralNetConfiguration.Builder().run(init).build()
}

fun NeuralNetConfiguration.Builder.layers(init: NeuralNetConfiguration.ListBuilder.() -> Unit): NeuralNetConfiguration.ListBuilder {
    return this.list().apply(init)
}

fun NeuralNetConfiguration.ListBuilder.dense(init: DenseLayer.Builder.() -> Unit) {
    this.layer(DenseLayer.Builder().apply(init).build())
}

fun NeuralNetConfiguration.ListBuilder.conv2d(init: ConvolutionLayer.Builder.() -> Unit) {
    this.layer(ConvolutionLayer.Builder().apply(init).build())
}

fun NeuralNetConfiguration.ListBuilder.output(init: OutputLayer.Builder.() -> Unit) {
    this.layer(OutputLayer.Builder().apply(init).build())
}