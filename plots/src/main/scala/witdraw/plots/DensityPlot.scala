package witdraw.plots

import plotwit.*

import viz.macros.VegaPlot
import viz.macros.SpecMod
import io.circe.Json

import dimwit.*

import scala.annotation.targetName

/** Kernel density estimates of samples, drawn as lines, in the style of
  * plotwit's `linePlot`: one density for a vector of samples, or one per row of
  * a matrix, e.g. one per MCMC chain.
  *
  * The density is estimated by Vega-Lite (its `density` transform, a Gaussian
  * kernel with Scott's bandwidth), so only the samples are sent to the viewer.
  * {{{
  * densityPlot(samples, _.title := "Posterior of sigma")
  * densityPlot(perChain, Seq("chain 0", "chain 1"), _.encoding.color.legend.title := "Chain")
  * }}}
  */
object densityPlot:

  lazy val template = new PlotTemplate(
    VegaPlot.fromString("""
      {
        "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
        "description": "Density plot",
        "title": "",
        "width": 300,
        "height": 200,
        "data": { "values": [] },
        "transform": [
          {
            "density": "value",
            "groupby": ["series"],
            "as": ["value", "density"]
          }
        ],
        "mark": { "type": "line", "strokeWidth": 2, "opacity": 1 },
        "encoding": {
          "x": {
            "field": "value",
            "type": "quantitative",
            "title": "Value",
            "scale": { "zero": false }
          },
          "y": {
            "field": "density",
            "type": "quantitative",
            "title": "Density"
          },
          "color": {
            "field": "series",
            "type": "nominal",
            "legend": { "title": "Series" }
          }
        }
      }""")
  )

  import template.{M, spec}

  /** The density of the samples in `data`. */
  def apply[S: Label](data: Tensor1[S, Float32], mods: (M => SpecMod)*): UnitSpec =
    build(Seq(None -> data.toArray), mods)

  /** One density per index of the L axis, of the samples along S. */
  @targetName("applySeries")
  def apply[L: Label, S: Label](data: Tensor2[L, S, Float32], mods: (M => SpecMod)*): UnitSpec =
    val label = summon[Label[L]].name
    apply(data, (0 until data.shape(Axis[L])).map(i => s"${label}_$i"), mods*)

  /** One named density per index of the L axis, of the samples along S. */
  @targetName("applyNamedSeries")
  def apply[L: Label, S: Label](
      data: Tensor2[L, S, Float32],
      names: Seq[String],
      mods: (M => SpecMod)*
  ): UnitSpec =
    val numSeries = data.shape(Axis[L])
    require(names.size == numSeries, s"Expected $numSeries series names, got ${names.size}")
    build(names.map(Some(_)).zip(data.toArray), mods)

  private def build(series: Seq[(Option[String], Array[Float])], mods: Seq[M => SpecMod]): UnitSpec = UnitSpec:
    val values = series.flatMap: (name, samples) =>
      samples.map: v =>
        val point = Json.obj("value" -> Json.fromFloatOrNull(v))
        name.fold(point)(n => point.deepMerge(Json.obj("series" -> Json.fromString(n))))
    // A single density has no series, so it needs neither a grouping nor a color.
    val single: Seq[M => SpecMod] =
      if series.forall(_._1.isEmpty) then
        Seq(
          _ =>
            json =>
              json.mapObject: obj =>
                obj
                  .add("transform", Json.arr(Json.obj(
                    "density" -> Json.fromString("value"),
                    "as" -> Json.arr(Json.fromString("value"), Json.fromString("density"))
                  )))
                  .add("encoding", obj("encoding").map(_.mapObject(_.remove("color"))).getOrElse(Json.obj()))
        )
      else Seq.empty
    spec.build((Seq[M => SpecMod](_.data.values := Json.fromValues(values)) ++ mods ++ single)*)
