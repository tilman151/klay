package org.klay.examples.utils

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.jfree.chart.ChartPanel
import org.jfree.chart.ChartUtilities
import org.jfree.chart.JFreeChart
import org.jfree.chart.axis.AxisLocation
import org.jfree.chart.axis.NumberAxis
import org.jfree.chart.block.BlockBorder
import org.jfree.chart.plot.DatasetRenderingOrder
import org.jfree.chart.plot.XYPlot
import org.jfree.chart.renderer.GrayPaintScale
import org.jfree.chart.renderer.PaintScale
import org.jfree.chart.renderer.xy.XYBlockRenderer
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer
import org.jfree.chart.title.PaintScaleLegend
import org.jfree.data.xy.*
import org.jfree.ui.RectangleEdge
import org.jfree.ui.RectangleInsets
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import java.awt.Color
import java.awt.Font
import java.util.*
import javax.swing.JFrame
import javax.swing.JPanel
import javax.swing.WindowConstants


/**
 * Simple plotting methods for the MLPClassifier quickstartexamples
 *
 * @author Alex Black
 */
object PlotUtil {
    /**
     * Plot the training data. Assume 2d input, classification output
     *
     * @param model         Model to use to get predictions
     * @param trainIter     DataSet Iterator
     * @param backgroundIn  sets of x,y points in input space, plotted in the background
     * @param nDivisions    Number of points (per axis, for the backgroundIn/backgroundOut arrays)
     */
    fun plotTrainingData(
        model: MultiLayerNetwork,
        trainIter: DataSetIterator,
        backgroundIn: INDArray,
        nDivisions: Int
    ) {
        val mins = backgroundIn.min(0).data().asDouble()
        val maxs = backgroundIn.max(0).data().asDouble()
        val ds = allBatches(trainIter)
        val backgroundOut = model.output(backgroundIn)
        val backgroundData: XYZDataset = createBackgroundData(backgroundIn, backgroundOut)
        val panel: JPanel =
            ChartPanel(createChart(backgroundData, mins, maxs, nDivisions, createDataSetTrain(ds.features, ds.labels)))
        val f = JFrame()
        f.add(panel)
        f.defaultCloseOperation = WindowConstants.EXIT_ON_CLOSE
        f.pack()
        f.title = "Training Data"
        f.isVisible = true
        f.setLocation(0, 0)
    }

    /**
     * Plot the training data. Assume 2d input, classification output
     *
     * @param model         Model to use to get predictions
     * @param testIter      Test Iterator
     * @param backgroundIn  sets of x,y points in input space, plotted in the background
     * @param nDivisions    Number of points (per axis, for the backgroundIn/backgroundOut arrays)
     */
    fun plotTestData(model: MultiLayerNetwork, testIter: DataSetIterator, backgroundIn: INDArray, nDivisions: Int) {
        val mins = backgroundIn.min(0).data().asDouble()
        val maxs = backgroundIn.max(0).data().asDouble()
        val backgroundOut = model.output(backgroundIn)
        val backgroundData: XYZDataset = createBackgroundData(backgroundIn, backgroundOut)
        val ds = allBatches(testIter)
        val predicted = model.output(ds.features)
        val panel: JPanel = ChartPanel(
            createChart(
                backgroundData,
                mins,
                maxs,
                nDivisions,
                createDataSetTest(ds.features, ds.labels, predicted)
            )
        )
        val f = JFrame()
        f.add(panel)
        f.defaultCloseOperation = WindowConstants.EXIT_ON_CLOSE
        f.pack()
        f.title = "Test Data"
        f.isVisible = true
        f.setLocationRelativeTo(null)
        //f.setLocation(100,100);
    }

    /**
     * Create data for the background data set
     */
    private fun createBackgroundData(backgroundIn: INDArray, backgroundOut: INDArray): XYZDataset {
        val nRows = backgroundIn.rows()
        val xValues = DoubleArray(nRows)
        val yValues = DoubleArray(nRows)
        val zValues = DoubleArray(nRows)
        for (i in 0 until nRows) {
            xValues[i] = backgroundIn.getDouble(i.toLong(), 0)
            yValues[i] = backgroundIn.getDouble(i.toLong(), 1)
            zValues[i] = backgroundOut.getDouble(i.toLong(), 0)
        }
        val dataset = DefaultXYZDataset()
        dataset.addSeries("Series 1", arrayOf(xValues, yValues, zValues))
        return dataset
    }

    //Training data
    private fun createDataSetTrain(features: INDArray, labels: INDArray): XYDataset {
        val nRows = features.rows()
        val nClasses = 2 // Binary classification using one output call end sigmoid.
        val series: Array<XYSeries?> = arrayOfNulls<XYSeries>(nClasses)
        for (i in series.indices) series[i] = XYSeries("Class $i")
        val argMax = Nd4j.getExecutioner().exec(IMax(labels, 1))
        for (i in 0 until nRows) {
            val classIdx = argMax.getDouble(i.toLong()).toInt()
            series[classIdx]?.add(features.getDouble(i.toLong(), 0), features.getDouble(i.toLong(), 1))
        }
        val c = XYSeriesCollection()
        for (s in series) c.addSeries(s)
        return c
    }

    //Test data
    private fun createDataSetTest(features: INDArray, labels: INDArray, predicted: INDArray): XYDataset {
        val nRows = features.rows()
        val nClasses = 2 // Binary classification using one output call end sigmoid.
        val series: Array<XYSeries?> = arrayOfNulls<XYSeries>(nClasses * nClasses)
        val series_index = intArrayOf(0, 3, 2, 1) //little hack to make the charts look consistent.
        for (i in 0 until nClasses * nClasses) {
            val trueClass = i / nClasses
            val predClass = i % nClasses
            val label = "actual=$trueClass, pred=$predClass"
            series[series_index[i]] = XYSeries(label)
        }
        val actualIdx = labels.argMax(1)
        val predictedIdx = predicted.argMax(1)
        for (i in 0 until nRows) {
            val classIdx = actualIdx.getInt(i)
            val predIdx = predictedIdx.getInt(i)
            val idx = series_index[classIdx * nClasses + predIdx]
            series[idx]?.add(features.getDouble(i.toLong(), 0), features.getDouble(i.toLong(), 1))
        }
        val c = XYSeriesCollection()
        for (s in series) c.addSeries(s)
        return c
    }

    private fun createChart(
        dataset: XYZDataset,
        mins: DoubleArray,
        maxs: DoubleArray,
        nPoints: Int,
        xyData: XYDataset
    ): JFreeChart {
        val xAxis = NumberAxis("X")
        xAxis.setRange(mins[0], maxs[0])
        val yAxis = NumberAxis("Y")
        yAxis.setRange(mins[1], maxs[1])
        val renderer = XYBlockRenderer()
        renderer.setBlockWidth((maxs[0] - mins[0]) / (nPoints - 1))
        renderer.setBlockHeight((maxs[1] - mins[1]) / (nPoints - 1))
        val scale: PaintScale = GrayPaintScale(0.0, 1.0)
        renderer.setPaintScale(scale)
        val plot = XYPlot(dataset, xAxis, yAxis, renderer)
        plot.setBackgroundPaint(Color.lightGray)
        plot.setDomainGridlinesVisible(false)
        plot.setRangeGridlinesVisible(false)
        plot.setAxisOffset(RectangleInsets(5.0, 5.0, 5.0, 5.0))
        val chart = JFreeChart("", plot)
        chart.getXYPlot().getRenderer().setSeriesVisibleInLegend(0, false)
        val scaleAxis = NumberAxis("Probability (class 1)")
        scaleAxis.setAxisLinePaint(Color.white)
        scaleAxis.setTickMarkPaint(Color.white)
        scaleAxis.setTickLabelFont(Font("Dialog", Font.PLAIN, 7))
        val legend = PaintScaleLegend(
            GrayPaintScale(),
            scaleAxis
        )
        legend.setStripOutlineVisible(false)
        legend.setSubdivisionCount(20)
        legend.setAxisLocation(AxisLocation.BOTTOM_OR_LEFT)
        legend.setAxisOffset(5.0)
        legend.setMargin(RectangleInsets(5.0, 5.0, 5.0, 5.0))
        legend.setFrame(BlockBorder(Color.red))
        legend.setPadding(RectangleInsets(10.0, 10.0, 10.0, 10.0))
        legend.setStripWidth(10.0)
        legend.setPosition(RectangleEdge.LEFT)
        chart.addSubtitle(legend)
        ChartUtilities.applyCurrentTheme(chart)
        plot.setDataset(1, xyData)
        val renderer2 = XYLineAndShapeRenderer()
        renderer2.setBaseLinesVisible(false)
        plot.setRenderer(1, renderer2)
        plot.setDatasetRenderingOrder(DatasetRenderingOrder.FORWARD)
        return chart
    }

    fun generatePointsOnGraph(xMin: Double, xMax: Double, yMin: Double, yMax: Double, nPointsPerAxis: Int): INDArray {
        //generate all the x,y points
        val evalPoints = Array(nPointsPerAxis * nPointsPerAxis) {
            DoubleArray(
                2
            )
        }
        var count = 0
        for (i in 0 until nPointsPerAxis) {
            for (j in 0 until nPointsPerAxis) {
                val x = i * (xMax - xMin) / (nPointsPerAxis - 1) + xMin
                val y = j * (yMax - yMin) / (nPointsPerAxis - 1) + yMin
                evalPoints[count][0] = x
                evalPoints[count][1] = y
                count++
            }
        }
        return Nd4j.create(evalPoints)
    }

    /**
     * This is to collect all the data and return it as one minibatch. Obviously only for use here with small datasets
     * @param iter
     * @return
     */
    private fun allBatches(iter: DataSetIterator): DataSet {
        val fullSet: MutableList<DataSet> = ArrayList()
        iter.reset()
        while (iter.hasNext()) {
            val miniBatchList = iter.next().asList()
            fullSet.addAll(miniBatchList)
        }
        iter.reset()
        return ListDataSetIterator(fullSet, fullSet.size).next()
    }
}