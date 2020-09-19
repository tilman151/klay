/** Adapted from https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/
 * deeplearning4j/examples/quickstart/modeling/variationalautoencoder/VaeMNIST2dPlots.java **/

package org.klay.examples.variationalautoencoder

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution
import org.deeplearning4j.nn.layers.variational.VariationalAutoencoder
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr
import org.deeplearning4j.optimize.api.BaseTrainingListener
import org.klay.examples.utils.VAEPlotUtil.MNISTLatentSpaceVisualizer
import org.klay.examples.utils.VAEPlotUtil.plotData
import org.klay.nn.layers
import org.klay.nn.sequential
import org.klay.nn.vae
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.learning.config.RmsProp
import org.slf4j.LoggerFactory
import java.io.IOException
import java.util.*


/**
 * A simple example of training a variational autoencoder on MNIST.
 * This example intentionally has a small hidden state Z (2 values) for visualization on a 2-grid.
 *
 * After training, this example plots 2 things:
 * 1. The MNIST digit reconstructions vs. the latent space
 * 2. The latent space values for the MNIST test set, as training progresses (every N minibatches)
 *
 * Note that for both plots, there is a slider at the top - change this to see how the reconstructions and latent
 * space changes over time.
 *
 * @author Alex Black
 */
object VaeMNIST2dPlots {
    private var visualize = true
    private val log = LoggerFactory.getLogger(VaeMNIST2dPlots::class.java)
    @Throws(IOException::class)
    @JvmStatic
    fun main(args: Array<String>) {
        val minibatchSize = 128
        val rngSeed = 12345
        val nEpochs = 20 //Total number of training epochs

        //Plotting configuration
        val plotEveryNMinibatches = 100 //Frequency with which to collect data for later plotting
        val plotMin = -5.0 //Minimum values for plotting (x and y dimensions)
        val plotMax = 5.0 //Maximum values for plotting (x and y dimensions)
        val plotNumSteps = 16 //Number of steps for reconstructions, between plotMin and plotMax

        //MNIST data for training
        val trainIter: DataSetIterator = MnistDataSetIterator(minibatchSize, true, rngSeed)

        //Neural net configuration
        Nd4j.getRandom().setSeed(rngSeed)
        val conf = sequential {
            seed(rngSeed.toLong())
            updater(RmsProp(1e-3))
            weightInit(WeightInit.XAVIER)
            l2(1e-4)
            layers {
                vae {
                    activation(Activation.LEAKYRELU)
                    encoderLayerSizes(256, 256)
                    decoderLayerSizes(256, 256)
                    pzxActivationFunction(Activation.IDENTITY)
                    reconstructionDistribution(BernoulliReconstructionDistribution(Activation.SIGMOID.activationFunction))
                    nIn(28 * 28)
                    nOut(2)
                }
            }
        }
        val net = MultiLayerNetwork(conf)
        net.init()

        //Get the variational autoencoder layer
        val vae = net.getLayer(0) as VariationalAutoencoder


        //Test data for plotting
        val testdata: DataSet = MnistDataSetIterator(10000, false, rngSeed).next()
        val testFeatures = testdata.features
        val testLabels = testdata.labels
        val latentSpaceGrid = getLatentSpaceGrid(plotMin, plotMax, plotNumSteps) //X/Y grid values, between plotMin and plotMax

        //Lists to store data for later plotting
        val latentSpaceVsEpoch: MutableList<INDArray> = ArrayList(nEpochs + 1)
        val latentSpaceValues = vae.activate(testFeatures, false, LayerWorkspaceMgr.noWorkspaces()) //Collect and record the latent space values before training starts
        latentSpaceVsEpoch.add(latentSpaceValues)
        val digitsGrid: MutableList<INDArray> = ArrayList()


        //Add a listener to the network that, every N=100 minibatches:
        // (a) collect the test set latent space values for later plotting
        // (b) collect the reconstructions at each point in the grid
        net.setListeners(PlottingListener(100, testFeatures, latentSpaceGrid, latentSpaceVsEpoch, digitsGrid))

        //Perform training
        for (i in 0 until nEpochs) {
            log.info("Starting epoch {} of {}", i + 1, nEpochs)
            net.pretrain(trainIter) //Note use of .pretrain(DataSetIterator) not fit(DataSetIterator) for unsupervised training
        }

        //plot by default
        if (visualize) {
            //Plot MNIST test set - latent space vs. iteration (every 100 minibatches by default)
            plotData(latentSpaceVsEpoch, testLabels, plotMin, plotMax, plotEveryNMinibatches)

            //Plot reconstructions - latent space vs. grid
            val imageScale = 2.0 //Increase/decrease this to zoom in on the digits
            val v = MNISTLatentSpaceVisualizer(imageScale, digitsGrid, plotEveryNMinibatches)
            v.visualize()
        }
    }

    //This simply returns a 2d grid: (x,y) for x=plotMin to plotMax, and y=plotMin to plotMax
    private fun getLatentSpaceGrid(plotMin: Double, plotMax: Double, plotSteps: Int): INDArray {
        val data = Nd4j.create(plotSteps * plotSteps, 2)
        val linspaceRow = Nd4j.linspace(plotMin, plotMax, plotSteps.toLong(), DataType.FLOAT)
        for (i in 0 until plotSteps) {
            data[NDArrayIndex.interval(i * plotSteps, (i + 1) * plotSteps), NDArrayIndex.point(0)].assign(linspaceRow)
            val yStart = plotSteps - i - 1
            data[NDArrayIndex.interval(yStart * plotSteps, (yStart + 1) * plotSteps), NDArrayIndex.point(1)].assign(linspaceRow.getDouble(i.toLong()))
        }
        return data
    }

    private class PlottingListener(private val plotEveryNMinibatches: Int, private val testFeatures: INDArray, private val latentSpaceGrid: INDArray,
                                   private val latentSpaceVsEpoch: MutableList<INDArray>, private val digitsGrid: MutableList<INDArray>) : BaseTrainingListener() {
        override fun iterationDone(model: Model, iterationCount: Int, epoch: Int) {
            if (model !is VariationalAutoencoder) {
                return
            }

            //Every N=100 minibatches:
            // (a) collect the test set latent space values for later plotting
            // (b) collect the reconstructions at each point in the grid
            if (iterationCount % plotEveryNMinibatches == 0) {
                val latentSpaceValues = model.activate(testFeatures, false, LayerWorkspaceMgr.noWorkspaces())
                latentSpaceVsEpoch.add(latentSpaceValues)
                val out = model.generateAtMeanGivenZ(latentSpaceGrid)
                digitsGrid.add(out)
            }
        }
    }
}