package witdraw.plots

import plotwit.*
import plotwit.plots.linePlot

import io.circe.Json

import dimwit.*
import dimwit.tensor.Labels

import witdraw.{SiteDef, SiteReader, SiteValues}
import witdraw.inference.{Diagnostics, SamplerInfo}

/** Trace plots of MCMC chains, in the style of ArviZ's `plot_trace`: for each
  * element of a quantity, one row with the density of every chain on the left
  * and the draws of every chain on the right, with R-hat and the effective
  * sample size in the row's title.
  *
  * It takes the chains of a sampler, and the sites to show, or values
  * computed from sites, made with [[quantity]]:
  * {{{
  * display(tracePlot(chains)(
  *   Sigma, Tau,
  *   quantity("beta", predictorNames)(v => shrunk(v(Z), v(Tau), v(Lambda)))
  * ))
  * }}}
  * The divergent transitions of NUTS are marked on the traces.
  *
  * For draws that do not come from a sampler, e.g. of a derived value, there
  * is a variant that takes the draws of one quantity, per chain.
  */
object TracePlot:

  /** The axes of the draws of one element: chain, and draw within the chain. */
  private trait Chain derives Label
  private trait Draw derives Label

  /** A value computed from the sites of a draw, to show in a trace plot; made
    * with [[quantity]].
    */
  final class Quantity private[TracePlot] (
      val name: String,
      private[TracePlot] val rows: (Seq[Seq[SiteValues]], Seq[Seq[Boolean]]) => Seq[VegaLiteSpec]
  )

  /** A value computed from the sites of each draw, e.g. a transformed
    * parameter.
    *
    * @param name
    *   the name of the value, shown in the row titles
    * @param elementNames
    *   one name per element of the value, e.g. the names of the predictors of
    *   a coefficient vector; by default `name[0]`, `name[1]`, ...
    */
  def quantity[T <: Tuple: Labels](name: String, elementNames: Seq[String] = Seq.empty)(
      f: SiteValues => Tensor[T, Float32]
  ): Quantity =
    Quantity(
      name,
      (chains, divergent) => rows(name, chains.map(_.map(f)), elementNames, divergent, diagnostics = true)
    )

  /** The trace plots of `quantities`, one below the other, from the chains of
    * a sampler, all of the same length. Divergent transitions are marked on
    * the traces.
    */
  def tracePlot[P, I](chains: Seq[witdraw.inference.Chain[P, I]])(
      quantities: (SiteDef[?, Float32] | Quantity)*
  )(using reader: SiteReader[P], info: SamplerInfo[I]): VegaLiteSpec =
    require(quantities.nonEmpty, "no quantities given")
    val values = chains.map(_.values(reader.values))
    val divergent = chains.map(_.divergent)
    val all = quantities
      .map:
        case q: Quantity                         => q
        case site: SiteDef[?, Float32] @unchecked => ofSite(site)
      .flatMap(_.rows(values, divergent))
    if all.size == 1 then all.head else vconcat(all)

  private def ofSite[T <: Tuple](site: SiteDef[T, Float32]): Quantity =
    quantity(site.name)(_(site))(using site.labels)

  /** The trace plot of one quantity, given as its draws: one sequence per
    * chain, all of the same length, as for [[Diagnostics.rhat]].
    *
    * @param name
    *   the name of the quantity, shown in the row titles
    * @param elementNames
    *   one name per element of the quantity; by default `name[0]`, `name[1]`, ...
    * @param divergent
    *   whether each draw was a divergent transition, as from
    *   `NUTSInfo.isDivergent`; divergent draws are marked on the trace
    * @param diagnostics
    *   whether to show R-hat and the effective sample size in the row titles
    */
  def tracePlot[T <: Tuple: Labels](
      name: String,
      chains: Seq[Seq[Tensor[T, Float32]]],
      elementNames: Seq[String] = Seq.empty,
      divergent: Seq[Seq[Boolean]] = Seq.empty,
      diagnostics: Boolean = true
  ): VegaLiteSpec =
    val all = rows(name, chains, elementNames, divergent, diagnostics)
    if all.size == 1 then all.head else vconcat(all)

  /** One row per element of the quantity. */
  private def rows[T <: Tuple: Labels](
      name: String,
      chains: Seq[Seq[Tensor[T, Float32]]],
      elementNames: Seq[String],
      divergent: Seq[Seq[Boolean]],
      diagnostics: Boolean
  ): Seq[VegaLiteSpec] =
    require(chains.nonEmpty, "no chains given")
    val numDraws = chains.head.size
    require(numDraws > 0, "the chains have no draws")
    require(
      chains.forall(_.size == numDraws),
      s"all chains must have the same number of draws, but have ${chains.map(_.size).distinct.mkString(", ")}"
    )
    require(
      divergent.isEmpty || (divergent.size == chains.size && divergent.forall(_.size == numDraws)),
      "divergent must have one flag per draw of each chain"
    )

    // values(c)(d)(e): chain c, draw d, element e
    val values = flatDraws(chains)
    val numElements = values.head.head.length
    require(
      elementNames.isEmpty || elementNames.size == numElements,
      s"${elementNames.size} element names for $numElements elements"
    )
    val names =
      if elementNames.nonEmpty then elementNames
      else if numElements == 1 then Seq(name)
      else (0 until numElements).map(e => s"$name[$e]")

    val (rhat, ess) =
      if diagnostics && numDraws >= 4 then
        (Some(flat(Diagnostics.rhat(chains))), Some(flat(Diagnostics.ess(chains))))
      else (None, None)

    val chainNames = chains.indices.map(c => s"chain $c")
    val drawIndex = Tensor1(Axis[Draw]).fromArray(Array.tabulate(numDraws)(_.toFloat))
    val divergentDraws =
      divergent.flatMap(_.zipWithIndex.collect { case (true, d) => d.toDouble }).distinct

    (0 until numElements).map: e =>
      val title = (rhat, ess) match
        case (Some(r), Some(n)) => f"${names(e)}   R-hat ${r(e)}%.3f · ESS ${n(e)}%.0f"
        case _                  => names(e)
      val perChain = Tensor2(Axis[Chain], Axis[Draw]).fromArray(
        values.map(_.map(_(e).toFloat).toArray).toArray
      )
      row(title, drawIndex, perChain, chainNames, divergentDraws)

  /** One row: the densities of the chains, and their traces with the
    * divergences on top.
    */
  private def row(
      title: String,
      drawIndex: Tensor1[Draw, Float32],
      perChain: Tensor2[Chain, Draw, Float32],
      chainNames: Seq[String],
      divergentDraws: Seq[Double]
  ): VegaLiteSpec =
    val density = densityPlot(
      perChain,
      chainNames,
      _.width := 180,
      _.height := 110,
      _.encoding.color.legend.title := "Chain"
    )
    val traces = linePlot(
      drawIndex,
      perChain,
      chainNames,
      _.mark += Json.obj("strokeWidth" -> Json.fromDoubleOrNull(0.7), "opacity" -> Json.fromDoubleOrNull(0.7)),
      _.encoding.x.title := "Draw",
      _.encoding.y.title := "Value",
      _.encoding.y += Json.obj("scale" -> Json.obj("zero" -> Json.False)),
      _.encoding.color.legend.title := "Chain"
    )
    val divergences = rulePlot(
      divergentDraws,
      _.mark.color := "black",
      _.mark.opacity := 0.5,
      _.encoding.x.title := "Draw"
    )
    val traceLayers = if divergentDraws.isEmpty then Seq(traces) else Seq(traces, divergences)
    HConcatSpec(
      specs = Seq(
        density,
        LayerSpec(
          traceLayers,
          sharedConfig = Some(Json.obj("width" -> Json.fromInt(480), "height" -> Json.fromInt(110)))
        )
      ),
      sharedConfig = Some(Json.obj("title" -> Json.fromString(title)))
    )

  /** The draws as nested arrays: chain, draw, flattened element. They are
    * stacked into one tensor and read at once, as reading them draw by draw
    * is slow.
    */
  private def flatDraws[T <: Tuple: Labels](
      chains: Seq[Seq[Tensor[T, Float32]]]
  ): Seq[Seq[Array[Double]]] =
    val all = flat(stack(chains.map(c => stack(c, Axis[Draw])), Axis[Chain]))
    val numChains = chains.size
    val numDraws = chains.head.size
    val numElements = all.length / (numChains * numDraws)
    Seq.tabulate(numChains, numDraws): (c, d) =>
      val start = (c * numDraws + d) * numElements
      all.slice(start, start + numElements)

  private def flat[T <: Tuple: Labels](t: Tensor[T, Float32]): Array[Double] =
    t.flatten.toArray.map(_.toDouble)
