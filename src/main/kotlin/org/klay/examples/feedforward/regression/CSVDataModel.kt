/** Adapted from https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-examples/src/main/java/org/
 * deeplearning4j/examples/quickstart/modeling/feedforward/regression/CSVDataModel.java **/

package org.klay.examples.feedforward.regression

import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.jfree.chart.ChartFactory
import org.jfree.chart.ChartPanel
import org.jfree.chart.plot.PlotOrientation
import org.jfree.data.xy.XYSeries
import org.jfree.data.xy.XYSeriesCollection
import org.klay.examples.utils.DownloaderUtility
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import java.io.File
import java.io.IOException
import java.util.*
import javax.swing.JFrame
import javax.swing.JPanel
import javax.swing.WindowConstants
import org.klay.nn.*


/**
 * Read a csv file. Fit and plot the data using Deeplearning4J.
 *
 * @author Robert Altena
 */
object CSVDataModel {
    var visualize = true
    var dataLocalPath: String? = null
    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {
        dataLocalPath = DownloaderUtility.DATAEXAMPLES.Download()
        val filename = File(dataLocalPath, "CSVPlotData.csv").absolutePath
        val ds = readCSVDataset(filename)
        val DataSetList = ArrayList<DataSet>()
        DataSetList.add(ds)
        plotDataset(DataSetList) //Plot the data, make sure we have the right data.
        val net = fitStraightline(ds)

        // Get the min and max x values, using Nd4j
        val preProcessor = NormalizerMinMaxScaler()
        preProcessor.fit(ds)
        val nSamples = 50
        val x =
            Nd4j.linspace(preProcessor.min.getInt(0).toLong(), preProcessor.max.getInt(0).toLong(), nSamples.toLong())
                .reshape(nSamples.toLong(), 1)
        val y = net.output(x)
        val modeloutput = DataSet(x, y)
        DataSetList.add(modeloutput)

        //plot on by default
        if (visualize) {
            plotDataset(DataSetList) //Plot data and model fit.
        }
    }

    /**
     * Fit a straight line using a neural network.
     *
     * @param ds The dataset to fit.
     * @return The network fitted to the data
     */
    private fun fitStraightline(ds: DataSet): MultiLayerNetwork {
        val seed = 12345
        val nEpochs = 200
        val learningRate = 0.00001
        val numInputs = 1
        val numOutputs = 1

        //
        // Hook up one input to the one output.
        // The resulting model is a straight line.
        //
        val conf = sequential {
            seed(seed.toLong())
            weightInit(WeightInit.XAVIER)
            updater(Nesterovs(learningRate, 0.9))
            layers {
                dense {
                    nIn(numInputs)
                    nOut(numOutputs)
                    activation(Activation.IDENTITY)
                }
                output {
                    lossFunction(LossFunction.MSE)
                    activation(Activation.IDENTITY)
                    nIn(numOutputs)
                    nOut(numOutputs)
                }
            }
        }
        val net = MultiLayerNetwork(conf)
        net.init()
        net.setListeners(ScoreIterationListener(1))
        for (i in 0 until nEpochs) {
            net.fit(ds)
        }
        return net
    }

    /**
     * Read a CSV file into a dataset.
     *
     *
     * Use the correct constructor:
     * DataSet ds = new RecordReaderDataSetIterator(rr,batchSize);
     * returns the data as follows:
     * ===========INPUT===================
     * [[12.89, 22.70],
     * [19.34, 20.47],
     * [16.94,  6.08],
     * [15.87,  8.42],
     * [10.71, 26.18]]
     *
     *
     * Which is not the way the framework likes its data.
     *
     *
     * This one:
     * RecordReaderDataSetIterator(rr,batchSize, 1, 1, true);
     * returns
     * ===========INPUT===================
     * [12.89, 19.34, 16.94, 15.87, 10.71]
     * =================OUTPUT==================
     * [22.70, 20.47,  6.08,  8.42, 26.18]
     *
     *
     * This can be used as is for regression.
     */
    @Throws(IOException::class, InterruptedException::class)
    private fun readCSVDataset(filename: String): DataSet {
        val batchSize = 1000
        val rr: RecordReader = CSVRecordReader()
        rr.initialize(FileSplit(File(filename)))
        val iter: DataSetIterator = RecordReaderDataSetIterator(rr, batchSize, 1, 1, true)
        return iter.next()
    }

    /**
     * Generate an xy plot of the datasets provided.
     */
    private fun plotDataset(DataSetList: ArrayList<DataSet>) {
        val c = XYSeriesCollection()
        val dscounter = 1 //use to name the dataseries
        for (ds in DataSetList) {
            val features = ds.features
            val outputs = ds.labels
            val nRows = features.rows()
            val series = XYSeries("S$dscounter")
            for (i in 0 until nRows) {
                series.add(features.getDouble(i.toLong()), outputs.getDouble(i.toLong()))
            }
            c.addSeries(series)
        }
        val title = "title"
        val xAxisLabel = "xAxisLabel"
        val yAxisLabel = "yAxisLabel"
        val orientation = PlotOrientation.VERTICAL
        val legend = false
        val tooltips = false
        val urls = false
        val chart =
            ChartFactory.createScatterPlot(title, xAxisLabel, yAxisLabel, c, orientation, legend, tooltips, urls)
        val panel: JPanel = ChartPanel(chart)
        val f = JFrame()
        f.add(panel)
        f.defaultCloseOperation = WindowConstants.EXIT_ON_CLOSE
        f.pack()
        f.title = "Training Data"
        f.isVisible = true
    }
}