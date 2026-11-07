package witdraw.plots

import plotwit.*

import viz.macros.VegaPlot
import viz.macros.SpecMod
import io.circe.Json

import dimwit.*

/** Vertical lines across the whole plot, one at each of the given x values.
  * Meant to be overlaid on another plot, to mark positions on its x axis, e.g.
  * the divergent draws of an MCMC chain on its trace:
  * {{{
  * overlay(linePlot(draws, values), rulePlot(divergentDraws, _.mark.color := "black"))
  * }}}
  * In an overlay, the plots share the x axis; give both the same x title, or
  * Vega-Lite shows both.
  */
object rulePlot:

  lazy val template = new PlotTemplate(
    VegaPlot.fromString("""
      {
        "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
        "description": "Rule plot",
        "title": "",
        "data": { "values": [] },
        "mark": {
          "type": "rule",
          "color": "#e45756",
          "strokeWidth": 1,
          "opacity": 0.6
        },
        "encoding": {
          "x": {
            "field": "x",
            "type": "quantitative",
            "title": "X"
          }
        }
      }""")
  )

  import template.{M, spec}

  /** A vertical line at each value of `xs`. */
  def apply[S: Label](xs: Tensor1[S, Float32], mods: (M => SpecMod)*): UnitSpec =
    apply(xs.toArray.toSeq.map(_.toDouble), mods*)

  /** A vertical line at each of `xs`, which may be empty, e.g. for no
    * divergences.
    */
  def apply(xs: Seq[Double], mods: (M => SpecMod)*): UnitSpec = UnitSpec:
    val values = xs.map(x => Json.obj("x" -> Json.fromDoubleOrNull(x)))
    spec.build((Seq[M => SpecMod](_.data.values := Json.fromValues(values)) ++ mods)*)
