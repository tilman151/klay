package org.klay.nn

import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder

fun NeuralNetConfiguration.ListBuilder.vae(init: VariationalAutoencoder.Builder.() -> Unit) {
    this.layer(VariationalAutoencoder.Builder().apply(init).build())
}

fun NeuralNetConfiguration.ListBuilder.embedding(init: EmbeddingLayer.Builder.() -> Unit) {
    this.layer(EmbeddingLayer.Builder().apply(init).build())
}