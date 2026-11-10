package witdraw.plots

import plotwit.*

import io.circe.Json

import dimwit.*
import dimwit.tensor.Labels

import dimwit.tensortree.{TensorTree, TreeOf}

import witdraw.{AxisNames, ModelResult}

/** Forest plots of MCMC chains, in the style of ArviZ's `plot_forest`: one row
  * per element of each quantity, with a point estimate and nested credible
  * intervals, the narrowest drawn thickest.
  *
  * It takes the chains of a sampler, and picks the quantities to show from the
  * model's result at each draw, as a named tuple; its fields name the rows:
  * {{{
  * display(forestPlot(chains, Axis[Predictor].names(predictorNames))(
  *   s => (sigma = s.sigma, beta = s.beta)
  * ))
  * }}}
  * By default the chains are pooled into one row per element, as with
  * `combined=True` in ArviZ, with 50% and 95% equal-tailed intervals around
  * the mean; [[Options]] changes that, e.g. one row per chain, in its own
  * color:
  * {{{
  * forestPlot(chains)(s => (sigma = s.sigma), Options(combined = false, ciProbs = Seq(0.8, 0.95)))
  * }}}
  */
object ForestPlot:

  /** The point drawn on each row. */
  enum PointEstimate:
    case Mean, Median

  /** The kind of the credible intervals: equal-tailed, between the quantiles
    * `(1 - p) / 2` and `(1 + p) / 2`, or the highest density interval, the
    * narrowest one holding a fraction `p` of the draws.
    */
  enum Interval:
    case ETI, HDI

  /** How to draw a forest plot.
    *
    * @param combined
    *   whether to pool the chains into one row per element, or show one row
    *   per chain
    * @param ciProbs
    *   the probabilities of the credible intervals; the narrowest is drawn
    *   thickest
    * @param pointEstimate
    *   the point drawn on each row
    * @param interval
    *   the kind of the credible intervals
    * @param width
    *   the width of the plot, in pixels
    */
  final case class Options(
      combined: Boolean = true,
      ciProbs: Seq[Double] = Seq(0.5, 0.95),
      pointEstimate: PointEstimate = PointEstimate.Mean,
      interval: Interval = Interval.ETI,
      width: Int = 500
  ):
    require(ciProbs.nonEmpty, "no credible interval probabilities given")
    require(
      ciProbs.forall(p => p > 0 && p < 1),
      s"credible interval probabilities must be between 0 and 1, but are ${ciProbs.mkString(", ")}"
    )

  /** The forest plot of the fields of `select`, from the chains of a sampler,
    * all of the same length, drawn as set in `options`.
    *
    * @param axisNames
    *   names for the entries along some axes, used in the row labels, e.g.
    *   `Axis[Predictor].names(predictorNames)`; by default the entries are
    *   numbered, `beta[0]`, `beta[1]`, ...
    * @param select
    *   the quantities to show, from the model's result at a draw
    */
  def forestPlot[P, I, A, R](
      chains: Seq[witdraw.inference.Chain[P, I]],
      axisNames: AxisNames[?]*
  )(using ModelResult[P, A])(select: A => R, options: Options = Options())(using
      TensorTree[R],
      TreeOf[R, Float32],
      FieldNames[R]
  ): VegaLiteSpec =
    plot(Draws.ofRecords(chains.map(_.samples.map(select)), axisNames), options)

  private def plot(quantities: Seq[Draws[?]], options: Options): VegaLiteSpec =
    import options.*
    val probs = ciProbs.distinct.sorted(using Ordering[Double].reverse)

    // One group of draws per row: label, chain (None when combined), draws.
    val groups = quantities.flatMap: draws =>
      val perDraw = draws.values
      draws.elementNames.zipWithIndex.flatMap: (label, e) =>
        val perChain = perDraw.map(_.map(_(e)).toArray)
        if combined then Seq((label, None, perChain.flatten.toArray))
        else perChain.zipWithIndex.map((xs, c) => (label, Some(s"chain $c"), xs))
    val labels = groups.map(_._1).distinct

    val rows = groups.map: (label, chain, xs) =>
      val sorted = xs.sorted
      val estimate = pointEstimate match
        case PointEstimate.Mean   => sorted.sum / sorted.length
        case PointEstimate.Median => quantile(sorted, 0.5)
      val intervals = probs.map: p =>
        interval match
          case Interval.ETI => (quantile(sorted, (1 - p) / 2), quantile(sorted, (1 + p) / 2))
          case Interval.HDI => hdi(sorted, p)
      (label, chain, estimate, intervals)

    def fields(label: String, chain: Option[String]): Seq[(String, Json)] =
      Seq("label" -> Json.fromString(label), "order" -> Json.fromInt(labels.indexOf(label))) ++
        chain.map(c => "chain" -> Json.fromString(c))

    val tooltipBase = Seq(Json.obj("field" -> Json.fromString("label"), "title" -> Json.fromString("Quantity"))) ++
      (if combined then Seq.empty else Seq(Json.obj("field" -> Json.fromString("chain"), "title" -> Json.fromString("Chain"))))

    def number(field: String, title: String): Json = Json.obj(
      "field" -> Json.fromString(field),
      "type" -> Json.fromString("quantitative"),
      "title" -> Json.fromString(title),
      "format" -> Json.fromString(".3g")
    )

    // the color of all marks when the chains are combined; with one row per
    // chain, the color encodes the chain instead
    val color = "#4c78a8"

    // the value axis, shared by all layers
    def xOf(field: String): Json = Json.obj(
      "field" -> Json.fromString(field),
      "type" -> Json.fromString("quantitative"),
      "title" -> Json.Null,
      "scale" -> Json.obj("zero" -> Json.False)
    )

    // the widest interval first, the thinnest line
    val intervalLayers = probs.zipWithIndex.map: (p, i) =>
      val data = rows.map: (label, chain, _, intervals) =>
        val (lo, hi) = intervals(i)
        Json.fromFields(fields(label, chain) ++ Seq("lo" -> Json.fromDoubleOrNull(lo), "hi" -> Json.fromDoubleOrNull(hi)))
      val percent = f"${p * 100}%.0f%%"
      UnitSpec(
        Json.obj(
          "data" -> Json.obj("values" -> Json.fromValues(data)),
          "mark" -> Json.obj(
            "type" -> Json.fromString("rule"),
            "color" -> Json.fromString(color),
            "strokeWidth" -> Json.fromDoubleOrNull(1.5 + 2.5 * i)
          ),
          "encoding" -> Json.obj(
            "x" -> xOf("lo"),
            "x2" -> Json.obj("field" -> Json.fromString("hi")),
            "tooltip" -> Json.fromValues(
              tooltipBase ++ Seq(number("lo", s"$percent lower"), number("hi", s"$percent upper"))
            )
          )
        )
      )

    val pointLayer = UnitSpec(
      Json.obj(
        "data" -> Json.obj(
          "values" -> Json.fromValues(rows.map: (label, chain, estimate, _) =>
            Json.fromFields(fields(label, chain) :+ ("estimate" -> Json.fromDoubleOrNull(estimate))))
        ),
        "mark" -> Json.obj(
          "type" -> Json.fromString("point"),
          "color" -> Json.fromString(color),
          "filled" -> Json.False,
          "fill" -> Json.fromString("white"),
          "fillOpacity" -> Json.fromInt(1),
          "strokeWidth" -> Json.fromDoubleOrNull(1.5),
          "size" -> Json.fromInt(40),
          "opacity" -> Json.fromInt(1)
        ),
        "encoding" -> Json.obj(
          "x" -> xOf("estimate"),
          "tooltip" -> Json.fromValues(tooltipBase :+ number("estimate", pointEstimate.toString))
        )
      )
    )

    val numChains = quantities.head.chains.size
    val step = if combined then 22 else math.max(22, 10 * numChains)
    val y = Json.obj(
      "field" -> Json.fromString("label"),
      "type" -> Json.fromString("nominal"),
      "sort" -> Json.obj("field" -> Json.fromString("order"), "op" -> Json.fromString("min")),
      "title" -> Json.Null
    )
    val perChain =
      if combined then Seq.empty
      else
        Seq(
          "yOffset" -> Json.obj("field" -> Json.fromString("chain"), "type" -> Json.fromString("nominal")),
          "color" -> Json.obj(
            "field" -> Json.fromString("chain"),
            "type" -> Json.fromString("nominal"),
            "legend" -> Json.obj("title" -> Json.fromString("Chain"))
          )
        )
    LayerSpec(
      intervalLayers :+ pointLayer,
      sharedConfig = Some(
        Json.obj(
          "width" -> Json.fromInt(width),
          "height" -> Json.obj("step" -> Json.fromInt(step)),
          "encoding" -> Json.fromFields(("y" -> y) +: perChain)
        )
      )
    )

  /** The `p` quantile of sorted values, interpolating linearly between them,
    * as NumPy does by default.
    */
  private def quantile(sorted: Array[Double], p: Double): Double =
    val pos = p * (sorted.length - 1)
    val lo = math.floor(pos).toInt
    val hi = math.min(lo + 1, sorted.length - 1)
    sorted(lo) + (pos - lo) * (sorted(hi) - sorted(lo))

  /** The narrowest interval holding a fraction `p` of the sorted values, as in
    * ArviZ.
    */
  private def hdi(sorted: Array[Double], p: Double): (Double, Double) =
    val n = sorted.length
    val inside = math.floor(p * n).toInt
    if inside <= 0 || inside >= n then (sorted.head, sorted.last)
    else
      val start = (0 until n - inside).minBy(i => sorted(i + inside) - sorted(i))
      (sorted(start), sorted(start + inside))
