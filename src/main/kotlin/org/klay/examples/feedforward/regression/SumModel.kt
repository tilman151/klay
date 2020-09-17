/** Adapted from https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/
 * deeplearning4j/examples/quickstart/modeling/feedforward/regression/SumModel.java **/

package org.klay.examples.feedforward.regression

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.util.*
import org.klay.nn.*


/**
 * Created by Anwar on 3/15/2016.
 * An example of regression neural network for performing addition
 */
object SumModel {
    //Random number generator seed, for reproducability
    const val seed = 12345

    //Number of epochs (full passes of the data)
    private const val nEpochs = 200

    //Number of data points
    private const val nSamples = 1000

    //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
    const val batchSize = 100

    //Network learning rate
    private const val learningRate = 0.01

    // The range of the sample data, data in range (0-1 is sensitive for NN, you can try other ranges and see how it effects the results
    // also try changing the range along with changing the activation function
    private const val MIN_RANGE = 0
    private const val MAX_RANGE = 3
    private val rng = Random(seed.toLong())
    @JvmStatic
    fun main(args: Array<String>) {

        //Generate the training data
        val iterator = getTrainingData(batchSize, rng)

        //Create the network
        val numInput = 2
        val numOutputs = 1
        val nHidden = 10
        val net = MultiLayerNetwork(
                sequential {
                    seed(SumModel.seed.toLong())
                    weightInit(WeightInit.XAVIER)
                    updater(Nesterovs(learningRate, 0.9))
                    layers {
                        dense {
                            nIn(numInput)
                            nOut(nHidden)
                            activation(Activation.TANH) //Change this to RELU and you will see the net learns very well very quickly
                        }
                        output {
                            lossFunction(LossFunctions.LossFunction.MSE)
                            activation(Activation.IDENTITY)
                            nIn(nHidden)
                            nOut(numOutputs)
                        }
                    }
                }
        )
        net.init()
        net.setListeners(ScoreIterationListener(1))


        //Train the network on the full data set, and evaluate in periodically
        for (i in 0 until nEpochs) {
            iterator.reset()
            net.fit(iterator)
        }
        // Test the addition of 2 numbers (Try different numbers here)
        val input = Nd4j.create(doubleArrayOf(0.111111, 0.3333333333333), 1, 2)
        val out = net.output(input, false)
        println(out)
    }

    private fun getTrainingData(batchSize: Int, rand: Random): DataSetIterator {
        val sum = DoubleArray(nSamples)
        val input1 = DoubleArray(nSamples)
        val input2 = DoubleArray(nSamples)
        for (i in 0 until nSamples) {
            input1[i] = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * rand.nextDouble()
            input2[i] = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * rand.nextDouble()
            sum[i] = input1[i] + input2[i]
        }
        val inputNDArray1 = Nd4j.create(input1, nSamples.toLong(), 1)
        val inputNDArray2 = Nd4j.create(input2, nSamples.toLong(), 1)
        val inputNDArray = Nd4j.hstack(inputNDArray1, inputNDArray2)
        val outPut = Nd4j.create(sum, nSamples.toLong(), 1)
        val dataSet = DataSet(inputNDArray, outPut)
        val listDs = dataSet.asList()
        listDs.shuffle(rng)
        return ListDataSetIterator(listDs, batchSize)
    }
}