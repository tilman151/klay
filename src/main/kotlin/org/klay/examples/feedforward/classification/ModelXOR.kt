/** Adapted from https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/
 * deeplearning4j/examples/quickstart/modeling/feedforward/classification/ModelXOR.java **/

package org.klay.examples.feedforward.classification

import org.deeplearning4j.nn.conf.distribution.UniformDistribution
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Sgd
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.LoggerFactory
import org.klay.nn.*


/**
 * This basic example shows how to manually create a DataSet and train it to an basic Network.
 *
 *
 * The network consists in 2 input-neurons, 1 hidden-layer with 4 hidden-neurons, and 2 output-neurons.
 *
 *
 * I choose 2 output neurons, (the first fires for false, the second fires for
 * true) because the Evaluation class needs one neuron per classification.
 *
 *
 * +---------+---------+---------------+--------------+
 * | Input 1 | Input 2 | Label 1(XNOR) | Label 2(XOR) |
 * +---------+---------+---------------+--------------+
 * |    0    |    0    |       1       |       0      |
 * +---------+---------+---------------+--------------+
 * |    1    |    0    |       0       |       1      |
 * +---------+---------+---------------+--------------+
 * |    0    |    1    |       0       |       1      |
 * +---------+---------+---------------+--------------+
 * |    1    |    1    |       1       |       0      |
 * +---------+---------+---------------+--------------+
 *
 * @author Peter Großmann
 * @author Dariusz Zbyrad
 */
object ModelXOR {
    private val log = LoggerFactory.getLogger(ModelXOR::class.java)
    @JvmStatic
    fun main(args: Array<String>) {
        val seed = 1234 // number used to initialize a pseudorandom number generator.
        val nEpochs = 10000 // number of training epochs
        log.info("Data preparation...")

        // list off input values, 4 training samples with data for 2
        // input-neurons each
        val input = Nd4j.zeros(4, 2)

        // correspondending list with expected output values, 4 training samples
        // with data for 2 output-neurons each
        val labels = Nd4j.zeros(4, 2)

        // create first dataset
        // when first input=0 and second input=0
        input.putScalar(intArrayOf(0, 0), 0)
        input.putScalar(intArrayOf(0, 1), 0)
        // then the first output fires for false, and the second is 0 (see class comment)
        labels.putScalar(intArrayOf(0, 0), 1)
        labels.putScalar(intArrayOf(0, 1), 0)

        // when first input=1 and second input=0
        input.putScalar(intArrayOf(1, 0), 1)
        input.putScalar(intArrayOf(1, 1), 0)
        // then xor is true, therefore the second output neuron fires
        labels.putScalar(intArrayOf(1, 0), 0)
        labels.putScalar(intArrayOf(1, 1), 1)

        // same as above
        input.putScalar(intArrayOf(2, 0), 0)
        input.putScalar(intArrayOf(2, 1), 1)
        labels.putScalar(intArrayOf(2, 0), 0)
        labels.putScalar(intArrayOf(2, 1), 1)

        // when both inputs fire, xor is false again - the first output should fire
        input.putScalar(intArrayOf(3, 0), 1)
        input.putScalar(intArrayOf(3, 1), 1)
        labels.putScalar(intArrayOf(3, 0), 1)
        labels.putScalar(intArrayOf(3, 1), 0)

        // create dataset object
        val ds = DataSet(input, labels)
        log.info("Network configuration and training...")
        val conf = sequential {
            updater(Sgd(0.1))
            seed(seed.toLong())
            biasInit(0.0) // init the bias with 0 - empirical value, too
            // The networks can process the input more quickly and more accurately by ingesting
            // minibatches 5-10 elements at a time in parallel.
            // This example runs better without, because the dataset is smaller than the mini batch size
            miniBatch(false)
            layers {
                dense {
                    nIn(2)
                    nOut(4)
                    activation(Activation.SIGMOID) // random initialize weights with values between 0 and 1
                    weightInit(UniformDistribution(0.0, 1.0))
                }
                output {
                    lossFunction(LossFunction.NEGATIVELOGLIKELIHOOD)
                    nOut(2)
                    activation(Activation.SOFTMAX)
                    weightInit(UniformDistribution(0.0, 1.0))
                }
            }
        }
        val net = MultiLayerNetwork(conf)
        net.init()

        // add an listener which outputs the error every 100 parameter updates
        net.setListeners(ScoreIterationListener(100))

        // C&P from LSTMCharModellingExample
        // Print the number of parameters in the network (and for each layer)
        println(net.summary())

        // here the actual learning takes place
        for (i in 0 until nEpochs) {
            net.fit(ds)
        }

        // create output for every training sample
        val output = net.output(ds.features)
        println(output)

        // let Evaluation prints stats how often the right output had the highest value
        val eval = Evaluation()
        eval.eval(ds.labels, output)
        println(eval.stats())
    }
}