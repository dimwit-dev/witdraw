package witdraw.plots

import plotwit.PlotTargets.websocket
import plotwit.*

import viz.macros.VegaPlot
import viz.macros.SpecMod
import io.circe.Json
import io.circe.syntax._

import dimwit.*

import scala.NamedTuple.NamedTuple
import scala.compiletime.constValueTuple

object BoxPlot:

  private lazy val boxTemplate = new PlotTemplate(
    VegaPlot.fromString("""
      {
        "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
        "description": "Box plot",
        "title": "",
        "data": { "values": [] },
        "mark": { "type": "boxplot", "extent": "min-max" },
        "encoding": {
          "x": {
            "field": "group",
            "type": "nominal",
            "title": "Group"
          },
          "y": {
            "field": "value",
            "type": "quantitative",
            "title": "Value"
          }
        }
      }""")
  )

  type TensorAxes[V <: Tuple] = Tuple.Union[V] match
    case dimwit.tensor.Tensor[t, Float32] => t

  type ElemLabel[V <: Tuple] = TensorAxes[V] match
    case Tuple1[s] => s

  /** Box plot over a named tuple of groups, eg
    * `boxplot((control = xs, treated = ys))`. The tuple's field names become
    * the group labels on the x-axis.
    */
  inline def boxplot[N <: Tuple, V <: Tuple](
      groups: NamedTuple[N, V],
      mods: (boxTemplate.M => SpecMod)*
  )(using Label[ElemLabel[V]]): UnitSpec = UnitSpec:
    val labels = constValueTuple[N].toList.asInstanceOf[List[String]]
    val tensors =
      groups.toTuple.toList.asInstanceOf[List[Tensor1[ElemLabel[V], Float32]]]

    val values = labels
      .zip(tensors)
      .flatMap: (label, tensor) =>
        tensor.toArray.map: v =>
          Json.obj(
            "group" -> label.asJson,
            "value" -> v.asJson
          )

    boxTemplate.spec.build(
      (Seq[boxTemplate.M => SpecMod](
        _.data.values := values.asJson
      ) ++ mods)*
    )
