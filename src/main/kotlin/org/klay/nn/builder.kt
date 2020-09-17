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

fun NeuralNetConfiguration.ListBuilder.batchNorm(init: BatchNormalization.Builder.() -> Unit) {
    this.layer(BatchNormalization.Builder().apply(init).build())
}

fun NeuralNetConfiguration.ListBuilder.subsampling(init: SubsamplingLayer.Builder.() -> Unit) {
    this.layer(SubsamplingLayer.Builder().apply(init).build())
}

fun NeuralNetConfiguration.ListBuilder.lstm(init: LSTM.Builder.() -> Unit) {
    this.layer(LSTM.Builder().apply(init).build())
}

fun NeuralNetConfiguration.ListBuilder.output(init: OutputLayer.Builder.() -> Unit) {
    this.layer(OutputLayer.Builder().apply(init).build())
}

fun NeuralNetConfiguration.ListBuilder.centerLossOutput(init: CenterLossOutputLayer.Builder.() -> Unit) {
    this.layer(CenterLossOutputLayer.Builder().apply(init).build())
}

fun NeuralNetConfiguration.ListBuilder.rnnOutput(init: RnnOutputLayer.Builder.() -> Unit) {
    this.layer(RnnOutputLayer.Builder().apply(init).build())
}