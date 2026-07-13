package witdraw
package plotting

import viz.vega.plots.SpecUrl
import io.github.quafadas.plots.SetupVega.{*, given}
import dimwit.*
import io.circe.syntax.*
import io.circe.literal.*
import viz.ChartLibrary.Vega

import viz.extensions.*
import viz.Utils
import viz.VizReturn
import dimwit.tensor.Tensor4
import viz.LowPriorityPlotTarget

object Plotting:

  type VegaJson = io.circe.Json

  sealed trait VegaLiteSpec

  case class UnitSpec(config: VegaJson) extends VegaLiteSpec

  case class LayerSpec(
      layer: List[VegaLiteSpec],
      sharedConfig: Option[VegaJson] = None
  ) extends VegaLiteSpec

  case class ConcatSpec(
      hconcat: Option[List[VegaLiteSpec]] = None,
      vconcat: Option[List[VegaLiteSpec]] = None,
      sharedConfig: Option[VegaJson] = None
  ) extends VegaLiteSpec

  case class VegaLiteRoot(
      spec: VegaLiteSpec,
      config: Option[VegaJson] = None,
      schema: String = "https://vega.github.io/schema/vega-lite/v6.json"
  )

  def overlay(plots: VegaLiteSpec*): LayerSpec =
    LayerSpec(plots.toList)

  def plotImage[L1: Label, L2: Label](img: Tensor2[L1, L2, Int32]): UnitSpec =
    val imgData = img.transpose.toArray
    val imagePlot = VegaPlot.fromString("""
      {
      "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
      "description": "Image created from data.",
      "data": { "values": [] },
      "mark": { "type": "rect" },
      "encoding": {
        "x": {"field": "x", "type": "ordinal", "axis": null},
        "y": {"field": "y", "type": "ordinal", "axis": null},
        "fill": {"field": "value", "type": "quantitative", "scale": {"scheme": "greys"}, "legend": null}
      }
    }
      """)
    UnitSpec(
      imagePlot
        .build(
          _.data.values := io.circe.Json.fromValues(
            imgData.zipWithIndex.flatMap { case (row, i) =>
              row.zipWithIndex.map { case (v, j) =>
                io.circe.Json.obj(
                  "x" -> io.circe.Json.fromInt(i),
                  "y" -> io.circe.Json.fromInt(j),
                  "value" -> io.circe.Json.fromFloatOrNull(v)
                )
              }
            }
          )
        )
    )

  def plotHistogram[L: Label](
      data: Tensor1[L, Float32],
      title: Option[String] = None
  ): UnitSpec =
    val histPlot = VegaPlot.fromString("""
      {
      "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
      "description": "Histogram created from data.",
      "title": "",
      "data": { "values": [] },
      "mark": { "type": "bar" },
      "encoding": {
        "x": {"field": "value", "type": "quantitative", "bin": {"maxbins": 20}},
        "y": {"aggregate": "count", "type": "quantitative"}
      }
    }
      """)
    UnitSpec(
      histPlot
        .build(
          _.title := title.getOrElse("Histogram"),
          _.data.values := io.circe.Json.fromValues(data.toArray.map { v =>
            io.circe.Json.obj(
              "value" -> io.circe.Json.fromFloatOrNull(v)
            )
          })
        )
    )

  def plotLine[L: Label](
      x: Tensor1[L, Float32],
      y: Tensor1[L, Float32]
  ): UnitSpec =
    val linePlot = VegaPlot.fromString("""
      {
      "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
      "description": "Line plot created from data.",
      "data": { "values": [] },
      "mark": { "type": "line" },
      "encoding": {
        "x": {"field": "x", "type": "quantitative"},
        "y": {"field": "y", "type": "quantitative"}
      }
    }
      """)
    UnitSpec(
      linePlot
        .build(
          _.data.values := io.circe.Json.fromValues(
            x.toArray.zip(y.toArray).map { case (xv, yv) =>
              io.circe.Json.obj(
                "x" -> io.circe.Json.fromFloatOrNull(xv),
                "y" -> io.circe.Json.fromFloatOrNull(yv)
              )
            }
          )
        )
    )

  def plotScatter[L: Label](
      x: Tensor1[L, Float32],
      y: Tensor1[L, Float32],
      title: Option[String] = None,
      xlabel: Option[String] = None,
      ylabel: Option[String] = None
  ): UnitSpec =
    val scatterPlot = VegaPlot.fromString("""
      {
      "$schema": "https://vega.github.io/schema/vega-lite/v6.json",      
      "description": "Scatter plot created from data.",
      "title": "",
      "data": { "values": [] },
      "mark": { "type": "point" },
      "encoding": {
        "x": {"field": "x", "type": "quantitative", "scale": {"zero": false}, "axis": { "title": "X Axis Label" }},
        "y": {"field": "y", "type": "quantitative", "scale": {"zero": false}, "axis": { "title": "Y Axis Label" }}
      }
    }
      """)
    UnitSpec(
      scatterPlot
        .build(
          _.title := title.getOrElse("Scatter Plot"),
          _.data.values := io.circe.Json
            .fromValues(x.toArray.zip(y.toArray).map { case (xv, yv) =>
              io.circe.Json.obj(
                "x" -> io.circe.Json.fromFloatOrNull(xv),
                "y" -> io.circe.Json.fromFloatOrNull(yv)
              )
            }),
          _.encoding.x.axis.title := xlabel.getOrElse("X Axis"),
          _.encoding.y.axis.title := ylabel.getOrElse("Y Axis")
        )
    )

  def emptyPlot: UnitSpec =
    val emptyPlot = VegaPlot.fromString("""
      {
      "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
      "description": "Empty plot.",
      "data": { "values": [] },
      "mark": { "type": "point" },
      "encoding": {
        "x": {"field": "x", "type": "quantitative"},
        "y": {"field": "y", "type": "quantitative"}
      }
    }
      """)
    UnitSpec(
      emptyPlot
        .build(
          _.data.values := io.circe.Json.fromValues(List.empty)
        )
    )

  def plotVectorField[X: Label, Y: Label, VecValue: Label](
      df: Tensor3[X, Y, VecValue, Float32]
  ): UnitSpec =
    val vectorFieldPlot = VegaPlot.fromString("""
      {
      "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
      "description": "Vector field plot created from data.",
      "data": { "values": [] },
      "layer": [
        {
          "mark": { "type": "rule" },
          "encoding": {
            "x":  {"field": "x",  "type": "quantitative"},
            "y":  {"field": "y",  "type": "quantitative"},
            "x2": {"field": "x2"},
            "y2": {"field": "y2"}
          }
        },
        {
          "transform": [
            {"calculate": "90 - atan2(datum.y2 - datum.y, datum.x2 - datum.x) * 180 / PI", "as": "angle"}
          ],
          "mark": { "type": "point", "shape": "triangle", "filled": true, "size": 30 },
          "encoding": {
            "x":     {"field": "x2",    "type": "quantitative"},
            "y":     {"field": "y2",    "type": "quantitative"},
            "angle": {"field": "angle", "type": "quantitative", "scale": {"domain": [-360, 360], "range": [-360, 360]}}
          }
        }
      ]
    }
      """)
    UnitSpec(
      vectorFieldPlot
        .build(
          _.data.values := io.circe.Json
            .fromValues(df.toArray.zipWithIndex.flatMap { case (row, i) =>
              row.zipWithIndex.map { case (vec, j) =>
                io.circe.Json.obj(
                  "x" -> io.circe.Json.fromFloatOrNull(i.toFloat),
                  "y" -> io.circe.Json.fromFloatOrNull(j.toFloat),
                  "x2" -> io.circe.Json.fromFloatOrNull(i.toFloat + vec(0)),
                  "y2" -> io.circe.Json.fromFloatOrNull(j.toFloat + vec(1))
                )
              }
            })
        )
    )

  private def stripSchema(json: VegaJson): VegaJson =
    json.mapObject(_.remove("$schema"))

  private def toJsonUnit(spec: VegaLiteSpec): VegaJson = spec match
    case UnitSpec(config)                => stripSchema(config)
    case LayerSpec(layers, sharedConfig) =>
      val base = io.circe.Json.obj(
        "layer" -> io.circe.Json.fromValues(layers.map(toJsonUnit))
      )
      sharedConfig.fold(base)(sc => sc.deepMerge(base))
    case ConcatSpec(hconcat, vconcat, sharedConfig) =>
      val fields = List(
        hconcat.map(hs =>
          "hconcat" -> io.circe.Json.fromValues(hs.map(toJsonUnit))
        ),
        vconcat.map(vs =>
          "vconcat" -> io.circe.Json.fromValues(vs.map(toJsonUnit))
        )
      ).flatten
      val base = io.circe.Json.obj(fields*)
      sharedConfig.fold(base)(sc => sc.deepMerge(base))

  private def toJsonRoot(spec: VegaLiteSpec): VegaJson = spec match
    case UnitSpec(config) => config
    case _                =>
      toJsonUnit(spec).mapObject(
        _.add(
          "$schema",
          io.circe.Json.fromString(
            "https://vega.github.io/schema/vega-lite/v6.json"
          )
        )
      )

  def grid(rows: List[List[VegaLiteSpec]]): ConcatSpec =
    ConcatSpec(vconcat = Some(rows.map(row => ConcatSpec(hconcat = Some(row)))))

  def display(spec: VegaLiteSpec)(using ev: LowPriorityPlotTarget): VizReturn =
    toJsonRoot(spec).plot()
