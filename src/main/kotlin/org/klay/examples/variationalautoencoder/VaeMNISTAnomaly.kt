package org.klay.examples.variationalautoencoder

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution
import org.deeplearning4j.nn.layers.variational.VariationalAutoencoder
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.klay.examples.feedforward.unsupervised.MNISTAutoencoder.MNISTVisualizer
import org.nd4j.common.primitives.Pair
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import java.io.IOException
import java.util.*
import kotlin.collections.HashMap
import org.klay.nn.*


/**
 * This example performs unsupervised anomaly detection on MNIST using a variational autoencoder, trained with a Bernoulli
 * reconstruction distribution.
 *
 * For details on the variational autoencoder, see:
 * - Kingma and Welling, 2013 - Auto-Encoding Variational Bayes - https://arxiv.org/abs/1312.6114
 *
 * For the use of VAEs for anomaly detection using reconstruction probability see:
 * - An & Cho, 2015 - Variational Autoencoder based Anomaly Detection using Reconstruction Probability
 * http://dm.snu.ac.kr/static/docs/TR/SNUDM-TR-2015-03.pdf
 *
 *
 * Unsupervised training is performed on the entire data set at once in this example. An alternative approach would be to
 * train one model for each digit.
 *
 * After unsupervised training, examples are scored using the VAE layer (reconstruction probability). Here, we are using the
 * labels to get the examples with the highest and lowest reconstruction probabilities for each digit for plotting. In a general
 * unsupervised anomaly detection situation, these labels would not be available, and hence highest/lowest probabilities
 * for the entire data set would be used instead.
 *
 * @author Alex Black
 */
object VaeMNISTAnomaly {
    private var visualize = true
    @Throws(IOException::class)
    @JvmStatic
    fun main(args: Array<String>) {
        val minibatchSize = 128
        val rngSeed = 12345
        val nEpochs = 5 //Total number of training epochs
        val reconstructionNumSamples = 16 //Reconstruction probabilities are estimated using Monte-Carlo techniques; see An & Cho for details

        //MNIST data for training
        val trainIter: DataSetIterator = MnistDataSetIterator(minibatchSize, true, rngSeed)

        //Neural net configuration
        Nd4j.getRandom().setSeed(rngSeed)
        val conf = sequential {
            seed(rngSeed.toLong())
            updater(Adam(1e-3))
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
                    nOut(32)
                }
            }
        }
        val net = MultiLayerNetwork(conf)
        net.init()
        net.setListeners(ScoreIterationListener(100))

        //Fit the data (unsupervised training)
        for (i in 0 until nEpochs) {
            net.pretrain(trainIter) //Note use of .pretrain(DataSetIterator) not fit(DataSetIterator) for unsupervised training
            println("Finished epoch " + (i + 1) + " of " + nEpochs)
        }


        //Perform anomaly detection on the test set, by calculating the reconstruction probability for each example
        //Then add pair (reconstruction probability, INDArray data) to lists and sort by score
        //This allows us to get best N and worst N digits for each digit type
        val testIter: DataSetIterator = MnistDataSetIterator(minibatchSize, false, rngSeed)

        //Get the variational autoencoder layer:
        val vae = net.getLayer(0) as VariationalAutoencoder
        val listsByDigit: MutableMap<Int, MutableList<Pair<Double, INDArray>>> = HashMap()
        for (i in 0..9) listsByDigit[i] = ArrayList()

        //Iterate over the test data, calculating reconstruction probabilities
        while (testIter.hasNext()) {
            val ds: DataSet = testIter.next()
            val features = ds.features
            val labels = Nd4j.argMax(ds.labels, 1) //Labels as integer indexes (from one hot), shape [minibatchSize, 1]
            val nRows = features.rows()

            //Calculate the log probability for reconstructions as per An & Cho
            //Higher is better, lower is worse
            val reconstructionErrorEachExample = vae.reconstructionLogProbability(features, reconstructionNumSamples) //Shape: [minibatchSize, 1]
            for (j in 0 until nRows) {
                val example = features.getRow(j.toLong(), true)
                val label = labels.getDouble(j.toLong()).toInt()
                val score = reconstructionErrorEachExample.getDouble(j.toLong())
                listsByDigit[label]!!.add(Pair(score, example))
            }
        }

        //Sort data by score, separately for each digit
        for (list in listsByDigit.values) {
            list.sortBy { -it.first }
        }

        //Select the 5 best and 5 worst numbers (by reconstruction probability) for each digit
        val best: MutableList<INDArray> = ArrayList(50)
        val worst: MutableList<INDArray> = ArrayList(50)
        val bestReconstruction: MutableList<INDArray> = ArrayList(50)
        val worstReconstruction: MutableList<INDArray> = ArrayList(50)
        for (i in 0..9) {
            val list: List<Pair<Double, INDArray>> = listsByDigit[i]!!
            for (j in 0..4) {
                val b = list[j].second
                val w = list[list.size - j - 1].second
                val mgr = LayerWorkspaceMgr.noWorkspaces()
                vae.setInput(b, mgr)
                val pzxMeanBest = vae.preOutput(false, mgr)
                val reconstructionBest = vae.generateAtMeanGivenZ(pzxMeanBest)
                vae.setInput(w, mgr)
                val pzxMeanWorst = vae.preOutput(false, mgr)
                val reconstructionWorst = vae.generateAtMeanGivenZ(pzxMeanWorst)
                best.add(b)
                bestReconstruction.add(reconstructionBest)
                worst.add(w)
                worstReconstruction.add(reconstructionWorst)
            }
        }

        //plot by default
        if (visualize) {
            //Visualize the best and worst digits
            val bestVisualizer = MNISTVisualizer(2.0, best, "Best (Highest Rec. Prob)")
            bestVisualizer.visualize()
            val bestReconstructions = MNISTVisualizer(2.0, bestReconstruction, "Best - Reconstructions")
            bestReconstructions.visualize()
            val worstVisualizer = MNISTVisualizer(2.0, worst, "Worst (Lowest Rec. Prob)")
            worstVisualizer.visualize()
            val worstReconstructions = MNISTVisualizer(2.0, worstReconstruction, "Worst - Reconstructions")
            worstReconstructions.visualize()
        }
    }
}