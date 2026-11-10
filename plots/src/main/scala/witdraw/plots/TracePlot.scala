package witdraw.plots

import plotwit.*
import plotwit.plots.linePlot

import io.circe.Json

import dimwit.*
import dimwit.tensor.Labels

import dimwit.tensortree.{TensorTree, TreeOf}

import witdraw.{AxisNames, ModelResult}
import witdraw.inference.{Diagnostics, SamplerInfo}

/** Trace plots of MCMC chains, in the style of ArviZ's `plot_trace`: for each
  * element of a quantity, one row with the density of every chain on the left
  * and the draws of every chain on the right, with R-hat and the effective
  * sample size in the row's title.
  *
  * It takes the chains of a sampler, and picks the quantities to show from the
  * model's result at each draw, as a named tuple; its fields name the rows:
  * {{{
  * display(tracePlot(chains)(s => (sigma = s.sigma, tau = s.tau)))
  * // with names for the entries along an axis: beta[male], beta[siblings], ...
  * display(tracePlot(chains, Axis[Predictor].names(predictorNames))(s => (beta = s.beta)))
  * }}}
  * The divergent transitions of NUTS are marked on the traces.
  */
object TracePlot:

  /** The axes of the draws of one element: chain, and draw within the chain. */
  private trait Chain derives Label
  private trait Draw derives Label

  /** The trace plots of the fields of `select`, one below the other, from the
    * chains of a sampler, all of the same length. Divergent transitions are
    * marked on the traces.
    *
    * @param axisNames
    *   names for the entries along some axes, e.g.
    *   `Axis[Predictor].names(predictorNames)`; by default the entries are
    *   numbered, `beta[0]`, `beta[1]`, ...
    * @param select
    *   the quantities to show, from the model's result at a draw
    */
  def tracePlot[P, I, A, R](
      chains: Seq[witdraw.inference.Chain[P, I]],
      axisNames: AxisNames[?]*
  )(using ModelResult[P, A], SamplerInfo[I])(select: A => R)(using
      TensorTree[R],
      TreeOf[R, Float32],
      FieldNames[R]
  ): VegaLiteSpec =
    val records = chains.map(_.samples.map(select))
    val divergent = chains.map(_.divergent)
    val all = Draws.ofRecords(records, axisNames).flatMap(rows(_, divergent, diagnostics = true))
    if all.size == 1 then all.head else vconcat(all)

  /** One row per element of the quantity. */
  private def rows[T <: Tuple](
      draws: Draws[T],
      divergent: Seq[Seq[Boolean]],
      diagnostics: Boolean
  ): Seq[VegaLiteSpec] =
    import draws.{chains, numDraws, numElements}
    given Labels[T] = draws.labels
    require(
      divergent.isEmpty || (divergent.size == chains.size && divergent.forall(_.size == numDraws)),
      "divergent must have one flag per draw of each chain"
    )

    // values(c)(d)(e): chain c, draw d, element e
    val values = draws.values
    val names = draws.elementNames

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

  private def flat[T <: Tuple: Labels](t: Tensor[T, Float32]): Array[Double] =
    Draws.flat(t)
