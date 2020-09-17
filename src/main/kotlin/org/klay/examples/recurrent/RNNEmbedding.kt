/** Adapted from https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/
 * deeplearning4j/examples/quickstart/modeling/recurrent/RNNEmbedding.java **/

package org.klay.examples.recurrent

import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.util.*
import org.klay.nn.*


/** Feed-forward layer that expects single integers per example as input (class numbers, in range 0 to numClass-1).
 * This input has shape [numExamples,1] instead of [numExamples,numClasses] for the equivalent one-hot representation.
 * Mathematically, EmbeddingLayer is equivalent to using a DenseLayer with a one-hot representation for the input; however,
 * it can be much more efficient with a large number of classes (as a dense layer + one-hot input does a matrix multiply
 * with all but one value being zero).<br></br>
 * **Note**: can only be used as the first layer for a network<br></br>
 * **Note 2**: For a given example index i, the output is activationFunction(weights.getRow(i) + bias), hence the
 * weight rows can be considered a vector/embedding for each example.
 *
 * @author Alex Black
 */
object RNNEmbedding {
    @JvmStatic
    fun main(args: Array<String>) {
        val nClassesIn = 10
        val batchSize = 3
        val timeSeriesLength = 8
        val inEmbedding = Nd4j.create(batchSize, 1, timeSeriesLength)
        val outLabels = Nd4j.create(batchSize, 4, timeSeriesLength)
        val r = Random(12345)
        for (i in 0 until batchSize) {
            for (j in 0 until timeSeriesLength) {
                val classIdx = r.nextInt(nClassesIn)
                inEmbedding.putScalar(intArrayOf(i, 0, j), classIdx)
                val labelIdx = r.nextInt(4)
                outLabels.putScalar(intArrayOf(i, labelIdx, j), 1.0)
            }
        }
        val conf = sequential {
            activation(Activation.RELU)
            layers {
                embedding {
                    nIn(nClassesIn)
                    nOut(5)
                }
                lstm {
                    nIn(5)
                    nOut(7)
                    activation(Activation.TANH)
                }
                rnnOutput {
                    lossFunction(LossFunctions.LossFunction.MCXENT)
                    nIn(7)
                    nOut(4)
                    activation(Activation.SOFTMAX)
                }
                inputPreProcessor(0, RnnToFeedForwardPreProcessor())
                inputPreProcessor(1, FeedForwardToRnnPreProcessor())
            }
        }
        val net = MultiLayerNetwork(conf)
        net.init()
        net.input = inEmbedding
        net.labels = outLabels
        net.computeGradientAndScore()
        println(net.score())
    }
}