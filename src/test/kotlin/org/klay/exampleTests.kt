package org.klay

import junit.framework.TestCase.assertEquals
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.weights.WeightInit
import org.junit.Test
import org.klay.nn.dense
import org.klay.nn.layers
import org.klay.nn.output
import org.klay.nn.sequential
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Nadam
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction


class ExampleTests {
    @Test
    fun mnistDoubleLayerExample() {
        val numRows = 28
        val numColumns = 28
        val outputNum = 10 // number of output classes
        val rngSeed = 123 // random number seed for reproducibility
        val rate = 0.0015 // learning rate

        val dl4jNet = NeuralNetConfiguration.Builder()
            .seed(rngSeed.toLong()) //include a random seed for reproducibility
            .activation(Activation.RELU)
            .weightInit(WeightInit.XAVIER)
            .updater(Nadam())
            .l2(rate * 0.005) // regularize learning model
            .list()
            .layer(
                DenseLayer.Builder() //create the first input layer.
                    .nIn(numRows * numColumns)
                    .nOut(500)
                    .build()
            )
            .layer(
                DenseLayer.Builder() //create the second input layer
                    .nIn(500)
                    .nOut(100)
                    .build()
            )
            .layer(
                OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                    .activation(Activation.SOFTMAX)
                    .nOut(outputNum)
                    .build()
            )
            .build()
        val dl4jString = dl4jNet.toString()

        val klayNet = sequential {
            seed(rngSeed.toLong()) //include a random seed for reproducibility
            // use stochastic gradient descent as an optimization algorithm

            activation(Activation.RELU)
            weightInit(WeightInit.XAVIER)
            updater(Nadam())
            l2(rate * 0.005) // regularize learning model
            layers {
                dense {
                    nIn(numRows * numColumns)
                    nOut(500)
                }
                dense {
                    nIn(500)
                    nOut(100)
                }
                output {
                    lossFunction(LossFunction.NEGATIVELOGLIKELIHOOD)
                    activation(Activation.SOFTMAX)
                    nIn(100)
                    nOut(outputNum)
                }
            }
        }
        val klayString = klayNet.toString()

        assertEquals(dl4jString, klayString)

    }
}