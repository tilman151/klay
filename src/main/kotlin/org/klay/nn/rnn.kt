package org.klay.nn

import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.LSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer

fun NeuralNetConfiguration.ListBuilder.lstm(init: LSTM.Builder.() -> Unit) {
    this.layer(LSTM.Builder().apply(init).build())
}

fun NeuralNetConfiguration.ListBuilder.rnnOutput(init: RnnOutputLayer.Builder.() -> Unit) {
    this.layer(RnnOutputLayer.Builder().apply(init).build())
}