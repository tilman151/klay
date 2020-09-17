package org.klay.examples.utils

import org.jfree.chart.ChartFactory
import org.jfree.chart.ChartPanel
import org.jfree.chart.JFreeChart
import org.jfree.chart.axis.NumberAxis
import org.jfree.chart.plot.PlotOrientation
import org.jfree.chart.plot.XYPlot
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer
import org.jfree.data.xy.XYDataset
import org.jfree.data.xy.XYSeries
import org.jfree.data.xy.XYSeriesCollection
import org.nd4j.common.primitives.Pair
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.awt.*
import java.awt.image.BufferedImage
import java.util.*
import javax.swing.*
import javax.swing.event.ChangeEvent
import javax.swing.event.ChangeListener


/**
 * Plotting methods for the VariationalAutoEncoder example
 * @author Alex Black
 */
object VAEPlotUtil {
    //Scatter plot util used for CenterLossMnistExample
    fun scatterPlot(data: List<Pair<INDArray, INDArray>>, epochCounts: List<Int>, title: String) {
        var xMin = Double.MAX_VALUE
        var xMax = -Double.MAX_VALUE
        var yMin = Double.MAX_VALUE
        var yMax = -Double.MAX_VALUE
        for (p in data) {
            val maxes = p.first.max(0)
            val mins = p.first.min(0)
            xMin = xMin.coerceAtMost(mins.getDouble(0))
            xMax = xMax.coerceAtLeast(maxes.getDouble(0))
            yMin = yMin.coerceAtMost(mins.getDouble(1))
            yMax = yMax.coerceAtLeast(maxes.getDouble(1))
        }
        val plotMin = xMin.coerceAtMost(yMin)
        val plotMax = xMax.coerceAtLeast(yMax)
        val panel: JPanel = ChartPanel(
            createChart(
                data[0].first,
                data[0].second,
                plotMin,
                plotMax,
                title + " (epoch " + epochCounts[0] + ")"
            )
        )
        val slider = JSlider(0, epochCounts.size - 1, 0)
        slider.snapToTicks = true
        val f = JFrame()
        slider.addChangeListener(object : ChangeListener {
            private var lastPanel: JPanel? = panel
            override fun stateChanged(e: ChangeEvent) {
                val jSlider = e.source as JSlider
                val value = jSlider.value
                val jPanel: JPanel = ChartPanel(
                    createChart(
                        data[value].first,
                        data[value].second,
                        plotMin,
                        plotMax,
                        title + " (epoch " + epochCounts[value] + ")"
                    )
                )
                if (lastPanel != null) {
                    f.remove(lastPanel)
                }
                lastPanel = jPanel
                f.add(jPanel, BorderLayout.CENTER)
                f.title = title
                f.revalidate()
            }
        })
        f.layout = BorderLayout()
        f.add(slider, BorderLayout.NORTH)
        f.add(panel, BorderLayout.CENTER)
        f.defaultCloseOperation = WindowConstants.EXIT_ON_CLOSE
        f.pack()
        f.title = title
        f.isVisible = true
    }

    fun plotData(xyVsIter: List<INDArray>, labels: INDArray, axisMin: Double, axisMax: Double, plotFrequency: Int) {
        val panel: JPanel = ChartPanel(createChart(xyVsIter[0], labels, axisMin, axisMax))
        val slider = JSlider(0, xyVsIter.size - 1, 0)
        slider.snapToTicks = true
        val f = JFrame()
        slider.addChangeListener(object : ChangeListener {
            private var lastPanel: JPanel? = panel
            override fun stateChanged(e: ChangeEvent) {
                val jSlider = e.source as JSlider
                val value = jSlider.value
                val jPanel: JPanel = ChartPanel(createChart(xyVsIter[value], labels, axisMin, axisMax))
                if (lastPanel != null) {
                    f.remove(lastPanel)
                }
                lastPanel = jPanel
                f.add(jPanel, BorderLayout.CENTER)
                f.title = getTitle(value, plotFrequency)
                f.revalidate()
            }
        })
        f.layout = BorderLayout()
        f.add(slider, BorderLayout.NORTH)
        f.add(panel, BorderLayout.CENTER)
        f.defaultCloseOperation = WindowConstants.EXIT_ON_CLOSE
        f.pack()
        f.title = getTitle(0, plotFrequency)
        f.isVisible = true
    }

    private fun getTitle(recordNumber: Int, plotFrequency: Int): String {
        return "MNIST Test Set - Latent Space Encoding at Training Iteration " + recordNumber * plotFrequency
    }

    //Test data
    private fun createDataSet(features: INDArray, labelsOneHot: INDArray): XYDataset {
        val nRows = features.rows()
        val nClasses = labelsOneHot.columns()
        val series = arrayOfNulls<XYSeries>(nClasses)
        for (i in 0 until nClasses) {
            series[i] = XYSeries(i.toString())
        }
        val classIdx = Nd4j.argMax(labelsOneHot, 1)
        for (i in 0 until nRows) {
            val idx = classIdx.getInt(i)
            series[idx]!!.add(features.getDouble(i.toLong(), 0), features.getDouble(i.toLong(), 1))
        }
        val c = XYSeriesCollection()
        for (s in series) c.addSeries(s)
        return c
    }

    private fun createChart(
        features: INDArray,
        labels: INDArray,
        axisMin: Double,
        axisMax: Double,
        title: String = "Variational Autoencoder Latent Space - MNIST Test Set"
    ): JFreeChart {
        val dataset = createDataSet(features, labels)
        val chart = ChartFactory.createScatterPlot(
            title,
            "X", "Y", dataset, PlotOrientation.VERTICAL, true, true, false
        )
        val plot = chart.plot as XYPlot
        plot.renderer.baseOutlineStroke = BasicStroke(0F)
        plot.noDataMessage = "NO DATA"
        plot.isDomainPannable = false
        plot.isRangePannable = false
        plot.isDomainZeroBaselineVisible = true
        plot.isRangeZeroBaselineVisible = true
        plot.domainGridlineStroke = BasicStroke(0.0f)
        plot.domainMinorGridlineStroke = BasicStroke(0.0f)
        plot.domainGridlinePaint = Color.blue
        plot.rangeGridlineStroke = BasicStroke(0.0f)
        plot.rangeMinorGridlineStroke = BasicStroke(0.0f)
        plot.rangeGridlinePaint = Color.blue
        plot.isDomainMinorGridlinesVisible = true
        plot.isRangeMinorGridlinesVisible = true
        val renderer = plot.renderer as XYLineAndShapeRenderer
        renderer.setSeriesOutlinePaint(0, Color.black)
        renderer.useOutlinePaint = true
        val domainAxis = plot.domainAxis as NumberAxis
        domainAxis.autoRangeIncludesZero = false
        domainAxis.setRange(axisMin, axisMax)
        domainAxis.tickMarkInsideLength = 2.0f
        domainAxis.tickMarkOutsideLength = 2.0f
        domainAxis.minorTickCount = 2
        domainAxis.isMinorTickMarksVisible = true
        val rangeAxis = plot.rangeAxis as NumberAxis
        rangeAxis.tickMarkInsideLength = 2.0f
        rangeAxis.tickMarkOutsideLength = 2.0f
        rangeAxis.minorTickCount = 2
        rangeAxis.isMinorTickMarksVisible = true
        rangeAxis.setRange(axisMin, axisMax)
        return chart
    }

    class MNISTLatentSpaceVisualizer(
        private val imageScale: Double, //Digits (as row vectors), one per INDArray
        private val digits: List<INDArray>, private val plotFrequency: Int
    ) {
        //Assume square, nxn rows
        private val gridWidth: Int = kotlin.math.sqrt(digits[0].size(0).toDouble()).toInt()
        private fun getTitle(recordNumber: Int): String {
            return "Reconstructions Over Latent Space at Training Iteration " + recordNumber * plotFrequency
        }

        fun visualize() {
            val frame = JFrame()
            frame.title = getTitle(0)
            frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
            frame.layout = BorderLayout()
            val panel = JPanel()
            panel.layout = GridLayout(0, gridWidth)
            val slider = JSlider(0, digits.size - 1, 0)
            slider.addChangeListener { e ->
                val jSlider = e.source as JSlider
                val value = jSlider.value
                panel.removeAll()
                val list = getComponents(value)
                for (image in list) {
                    panel.add(image)
                }
                frame.title = getTitle(value)
                frame.revalidate()
            }
            frame.add(slider, BorderLayout.NORTH)
            val list = getComponents(0)
            for (image in list) {
                panel.add(image)
            }
            frame.add(panel, BorderLayout.CENTER)
            frame.isVisible = true
            frame.pack()
        }

        private fun getComponents(idx: Int): List<JLabel> {
            val images: MutableList<JLabel> = ArrayList()
            val temp: MutableList<INDArray> = ArrayList()
            for (i in 0 until digits[idx].size(0)) {
                temp.add(digits[idx].getRow(i))
            }
            for (arr in temp) {
                val bi = BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY)
                for (i in 0..783) {
                    bi.raster.setSample(i % 28, i / 28, 0, (255 * arr.getDouble(i.toLong())).toInt())
                }
                val orig = ImageIcon(bi)
                val imageScaled = orig.image.getScaledInstance(
                    (imageScale * 28).toInt(),
                    (imageScale * 28).toInt(), Image.SCALE_REPLICATE
                )
                val scaled = ImageIcon(imageScaled)
                images.add(JLabel(scaled))
            }
            return images
        }
    }
}