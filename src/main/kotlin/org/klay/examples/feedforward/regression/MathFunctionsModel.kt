/** Adapted from https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/
 * deeplearning4j/examples/quickstart/modeling/feedforward/regression/MathFunctionsModel.java **/

package org.klay.examples.feedforward.regression

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.conf.MultiLayerConfiguration
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.jfree.chart.ChartFactory
import org.jfree.chart.ChartPanel
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.xy.XYSeries
import org.jfree.data.xy.XYSeriesCollection
import org.nd4j.linalg.ops.transforms.Transforms
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.util.*
import javax.swing.JFrame
import javax.swing.WindowConstants
import kotlin.collections.ArrayList
import org.klay.nn.*


/**Example: Train a network to reproduce certain mathematical functions, and plot the results.
 * Plotting of the network output occurs every 'plotFrequency' epochs. Thus, the plot shows the accuracy of the network
 * predictions as training progresses.
 * A number of mathematical functions are implemented here.
 * Note the use of the identity function on the network output layer, for regression
 *
 * @author Alex Black
 */
object MathFunctionsModel {
    private const val visualize = true

    //Random number generator seed, for reproducability
    const val seed = 12345

    //Number of epochs (full passes of the data)
    private const val nEpochs = 2000

    //How frequently should we plot the network output?
    private const val plotFrequency = 500

    //Number of data points
    private const val nSamples = 1000

    //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
    const val batchSize = 100

    //Network learning rate
    private const val learningRate = 0.01
    private val rng = Random(seed.toLong())
    private const val numInputs = 1
    private const val numOutputs = 1
    @JvmStatic
    fun main(args: Array<String>) {

        //Switch these two options to do different functions with different networks
        val fn: MathFunction = SinXDivXMathFunction()
        val conf = deepDenseLayerNetworkConfiguration

        //Generate the training data
        val x = Nd4j.linspace(-10, 10, nSamples.toLong()).reshape(nSamples.toLong(), 1)
        val iterator = getTrainingData(x, fn, batchSize, rng)

        //Create the network
        val net = MultiLayerNetwork(conf)
        net.init()
        net.setListeners(ScoreIterationListener(1))


        //Train the network on the full data set, and evaluate in periodically
        val networkPredictions = ArrayList<INDArray>()
        for (i in 0 until nEpochs) {
            iterator.reset()
            net.fit(iterator)
            if ((i + 1) % plotFrequency == 0) {
                networkPredictions.add(net.output(x, false))
            }
        }

        //Plots the target data and the network predictions by default
        if (visualize) {
            plot(fn, x, fn.getFunctionValues(x), networkPredictions)
        }
    }

    /** Returns the network configuration, 2 hidden DenseLayers of size 50.
     */
    private val deepDenseLayerNetworkConfiguration: MultiLayerConfiguration
        get() {
            val numHiddenNodes = 50
            return sequential {
                seed(MathFunctionsModel.seed.toLong())
                weightInit(WeightInit.XAVIER)
                activation(Activation.TANH)
                updater(Nesterovs(learningRate, 0.9))
                layers {
                    dense {
                        nIn(numInputs)
                        nOut(numHiddenNodes)
                    }
                    dense {
                        nIn(numHiddenNodes)
                        nOut(numHiddenNodes)
                    }
                    output {
                        lossFunction(LossFunctions.LossFunction.MSE)
                        activation(Activation.IDENTITY)
                        nIn(numHiddenNodes)
                        nOut(numOutputs)
                    }
                }
            }
        }

    /** Create a DataSetIterator for training
     * @param x X values
     * @param function Function to evaluate
     * @param batchSize Batch size (number of quickstartexamples for every call of DataSetIterator.next())
     * @param rng Random number generator (for repeatability)
     */
    private fun getTrainingData(x: INDArray, function: MathFunction, batchSize: Int, rng: Random): DataSetIterator {
        val y: INDArray = function.getFunctionValues(x)
        val allData = DataSet(x, y)
        val list = allData.asList()
        list.shuffle(rng)
        return ListDataSetIterator(list, batchSize)
    }

    //Plot the data
    private fun plot(function: MathFunction, x: INDArray, y: INDArray, predicted: ArrayList<INDArray>) {
        val dataSet = XYSeriesCollection()
        addSeries(dataSet, x, y, "True Function (Labels)")
        for (i in predicted.indices) {
            addSeries(dataSet, x, predicted[i], i.toString())
        }
        val chart = ChartFactory.createXYLineChart(
                "Regression Example - " + function.name,  // chart title
                "X", function.name + "(X)",  // y axis label
                dataSet,  // data
                PlotOrientation.VERTICAL,
                true,  // include legend
                true,  // tooltips
                false // urls
        )
        val panel = ChartPanel(chart)
        val f = JFrame()
        f.add(panel)
        f.defaultCloseOperation = WindowConstants.EXIT_ON_CLOSE
        f.pack()
        f.isVisible = true
    }

    private fun addSeries(dataSet: XYSeriesCollection, x: INDArray, y: INDArray, label: String) {
        val xd = x.data().asDouble()
        val yd = y.data().asDouble()
        val s = XYSeries(label)
        for (j in xd.indices) s.add(xd[j], yd[j])
        dataSet.addSeries(s)
    }
}

interface MathFunction {
    fun getFunctionValues(x: INDArray): INDArray
    val name: String
}

class SinXDivXMathFunction : MathFunction {
    override fun getFunctionValues(x: INDArray): INDArray {
        return Transforms.sin(x, true).divi(x)
    }

    override val name: String
        get() = "SinXDivX"
}