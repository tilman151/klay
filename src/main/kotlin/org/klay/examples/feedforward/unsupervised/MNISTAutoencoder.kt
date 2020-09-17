/** Adapted from https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/
 * deeplearning4j/examples/quickstart/modeling/feedforward/unsupervised/MNISTAutoencoder.java **/

package org.klay.examples.feedforward.unsupervised

import org.apache.commons.lang3.tuple.ImmutablePair
import org.apache.commons.lang3.tuple.Pair
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.AdaGrad
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import java.awt.GridLayout
import java.awt.Image
import java.awt.image.BufferedImage
import java.util.*
import javax.swing.ImageIcon
import javax.swing.JFrame
import javax.swing.JLabel
import javax.swing.JPanel
import kotlin.collections.HashMap
import org.klay.nn.*


/**Example: Anomaly Detection on MNIST using simple autoencoder without pretraining
 * The goal is to identify outliers digits, i.e., those digits that are unusual or
 * not like the typical digits.
 * This is accomplished in this example by using reconstruction error: stereotypical
 * examples should have low reconstruction error, whereas outliers should have high
 * reconstruction error. The number of epochs here is set to 3. Set to 30 for even better
 * results.
 *
 * @author Alex Black
 */
object MNISTAutoencoder {
    private const val visualize = true
    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {

        //Set up network. 784 in/out (as MNIST images are 28x28).
        //784 -> 250 -> 10 -> 250 -> 784
        val hiddenUnits = listOf(784, 250, 10, 250)
        val outputUnits = 784
        val conf = sequential {
            seed(12345)
            weightInit(WeightInit.XAVIER)
            updater(AdaGrad(0.05))
            activation(Activation.RELU)
            l2(0.0001)
            layers {
                for (u in hiddenUnits.zipWithNext()) { // Dynamically generate layers from units list
                    dense {
                        nIn(u.first)
                        nOut(u.second)
                    }
                }
                output {
                    lossFunction(LossFunction.MSE)
                    nIn(hiddenUnits.last())
                    nOut(outputUnits)
                }
            }
        }
        val net = MultiLayerNetwork(conf)
        net.listeners = listOf(ScoreIterationListener(10))

        //Load data and split into training and testing sets. 40000 train, 10000 test
        val iter: DataSetIterator = MnistDataSetIterator(100, 50000, false)
        val featuresTrain: MutableList<INDArray> = ArrayList()
        val featuresTest: MutableList<INDArray> = ArrayList()
        val labelsTest: MutableList<INDArray> = ArrayList()
        val r = Random(12345)
        while (iter.hasNext()) {
            val ds = iter.next()
            val split = ds.splitTestAndTrain(80, r) //80/20 split (from miniBatch = 100)
            featuresTrain.add(split.train.features)
            val dsTest = split.test
            featuresTest.add(dsTest.features)
            val indexes = Nd4j.argMax(dsTest.labels, 1) //Convert from one-hot representation -> index
            labelsTest.add(indexes)
        }

        //Train model:
        val nEpochs = 3
        for (epoch in 0 until nEpochs) {
            for (data in featuresTrain) {
                net.fit(data, data)
            }
            println("Epoch $epoch complete")
        }

        //Evaluate the model on the test data
        //Score each example in the test set separately
        //Compose a map that relates each digit to a list of (score, example) pairs
        //Then find N best and N worst scores per digit
        val listsByDigit: MutableMap<Int, MutableList<Pair<Double, INDArray>>> = HashMap()
        for (i in 0..9) listsByDigit[i] = ArrayList()
        for (i in featuresTest.indices) {
            val testData = featuresTest[i]
            val labels = labelsTest[i]
            val nRows = testData.rows()
            for (j in 0 until nRows) {
                val example = testData.getRow(j.toLong(), true)
                val digit = labels.getDouble(j.toLong()).toInt()
                val score = net.score(DataSet(example, example))
                // Add (score, example) pair to the appropriate list
                listsByDigit[digit]!!.add(ImmutablePair(score, example))
            }
        }

        //Sort each list in the map by score
        for (digitAllPairs in listsByDigit.values) {
            digitAllPairs.sortBy { it.left }
        }

        //After sorting, select N best and N worst scores (by reconstruction error) for each digit, where N=5
        val best: MutableList<INDArray> = ArrayList(50)
        val worst: MutableList<INDArray> = ArrayList(50)
        for (i in 0..9) {
            val list: List<Pair<Double, INDArray>> = listsByDigit[i]!!
            for (j in 0..4) {
                best.add(list[j].right)
                worst.add(list[list.size - j - 1].right)
            }
        }

        //Visualize by default
        if (visualize) {
            //Visualize the best and worst digits
            val bestVisualizer = MNISTVisualizer(2.0, best, "Best (Low Rec. Error)")
            bestVisualizer.visualize()
            val worstVisualizer = MNISTVisualizer(2.0, worst, "Worst (High Rec. Error)")
            worstVisualizer.visualize()
        }
    }

    class MNISTVisualizer @JvmOverloads constructor(private val imageScale: Double, //Digits (as row vectors), one per INDArray
                                                    private val digits: List<INDArray>,
                                                    private val title: String,
                                                    private val gridWidth: Int = 5) {
        fun visualize() {
            val frame = JFrame()
            frame.title = title
            frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
            val panel = JPanel()
            panel.layout = GridLayout(0, gridWidth)
            val list = components
            for (image in list) {
                panel.add(image)
            }
            frame.add(panel)
            frame.isVisible = true
            frame.pack()
        }

        private val components: List<JLabel>
            get() {
                val images: MutableList<JLabel> = ArrayList()
                for (arr in digits) {
                    val bi = BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY)
                    for (i in 0..783) {
                        bi.raster.setSample(i % 28, i / 28, 0, (255 * arr.getDouble(i.toLong())).toInt())
                    }
                    val orig = ImageIcon(bi)
                    val imageScaled = orig.image.getScaledInstance((imageScale * 28).toInt(), (imageScale * 28).toInt(), Image.SCALE_REPLICATE)
                    val scaled = ImageIcon(imageScaled)
                    images.add(JLabel(scaled))
                }
                return images
            }

    }
}