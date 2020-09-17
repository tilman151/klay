/** Adapted from https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/
 * deeplearning4j/examples/quickstart/modeling/recurrent/MemorizeSequence.java **/

package org.klay.examples.recurrent

import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.LSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.klay.nn.layers
import org.klay.nn.lstm
import org.klay.nn.rnnOutput
import org.klay.nn.sequential
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.RmsProp
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.util.*
import kotlin.collections.LinkedHashSet
import kotlin.collections.MutableList


/**
 * This example trains a RNN. When trained we only have to put the first
 * character of LEARNSTRING to the RNN, and it will recite the following chars
 *
 * @author Peter Grossmann
 */
object MemorizeSequence {
    // define a sentence to learn.
    // Add a special character at the beginning so the RNN learns the complete string and ends with the marker.
    private val LEARNSTRING = "*Der Cottbuser Postkutscher putzt den Cottbuser Postkutschkasten.".toCharArray()

    // a list of all possible characters
    private val LEARNSTRING_CHARS_LIST: MutableList<Char> = ArrayList()

    // RNN dimensions
    private const val HIDDEN_LAYER_WIDTH = 50
    private const val HIDDEN_LAYER_CONT = 2
    @JvmStatic
    fun main(args: Array<String>) {

        // create a dedicated list of possible chars in LEARNSTRING_CHARS_LIST
        val LEARNSTRING_CHARS = LinkedHashSet<Char>()
        for (c in LEARNSTRING) LEARNSTRING_CHARS.add(c)
        LEARNSTRING_CHARS_LIST.addAll(LEARNSTRING_CHARS)

        // some common parameters
        val conf = sequential {
            seed(123)
            biasInit(0.0)
            miniBatch(false)
            updater(RmsProp(0.001))
            weightInit(WeightInit.XAVIER)
            layers {
                for (i in 0 until HIDDEN_LAYER_CONT) {
                    lstm {
                        nIn(if (i == 0) LEARNSTRING_CHARS.size else HIDDEN_LAYER_WIDTH)
                        nOut(HIDDEN_LAYER_WIDTH)
                        activation(Activation.TANH)
                    }
                }
                rnnOutput {
                    lossFunction(LossFunctions.LossFunction.MCXENT)
                    activation(Activation.SOFTMAX)
                    nIn(HIDDEN_LAYER_WIDTH)
                    nOut(LEARNSTRING_CHARS.size)
                }
            }
        }
        val net = MultiLayerNetwork(conf)
        net.init()
        net.setListeners(ScoreIterationListener(1))

        /*
		 * CREATE OUR TRAINING DATA
		 */
        // create input and output arrays: SAMPLE_INDEX, INPUT_NEURON,
        // SEQUENCE_POSITION
        val input = Nd4j.zeros(1, LEARNSTRING_CHARS_LIST.size, LEARNSTRING.size)
        val labels = Nd4j.zeros(1, LEARNSTRING_CHARS_LIST.size, LEARNSTRING.size)
        // loop through our sample-sentence
        for ((samplePos, currentChar) in LEARNSTRING.withIndex()) {
            // small hack: when currentChar is the last, take the first char as
            // nextChar - not really required. Added to this hack by adding a starter first character.
            val nextChar = LEARNSTRING[(samplePos + 1) % LEARNSTRING.size]
            // input neuron for current-char is 1 at "samplePos"
            input.putScalar(intArrayOf(0, LEARNSTRING_CHARS_LIST.indexOf(currentChar), samplePos), 1)
            // output neuron for next-char is 1 at "samplePos"
            labels.putScalar(intArrayOf(0, LEARNSTRING_CHARS_LIST.indexOf(nextChar), samplePos), 1)
        }
        val trainingData = DataSet(input, labels)

        // some epochs
        for (epoch in 0..999) {
            println("Epoch $epoch")

            // train the data
            net.fit(trainingData)

            // clear current stance from the last example
            net.rnnClearPreviousState()

            // put the first character into the rrn as an initialisation
            val testInit = Nd4j.zeros(1, LEARNSTRING_CHARS_LIST.size, 1)
            testInit.putScalar(LEARNSTRING_CHARS_LIST.indexOf(LEARNSTRING[0]).toLong(), 1)

            // run one step -> IMPORTANT: rnnTimeStep() must be called, not
            // output()
            // the output shows what the net thinks what should come next
            var output = net.rnnTimeStep(testInit)

            // now the net should guess LEARNSTRING.length more characters
            for (ignored in LEARNSTRING) {

                // first process the last output of the network to a concrete
                // neuron, the neuron with the highest output has the highest
                // chance to get chosen
                val sampledCharacterIdx = Nd4j.getExecutioner().exec(IMax(output, 1)).getInt(0)

                // print the chosen output
                print(LEARNSTRING_CHARS_LIST[sampledCharacterIdx])

                // use the last output as input
                val nextInput = Nd4j.zeros(1, LEARNSTRING_CHARS_LIST.size, 1)
                nextInput.putScalar(sampledCharacterIdx.toLong(), 1)
                output = net.rnnTimeStep(nextInput)
            }
            print("\n")
        }
    }
}