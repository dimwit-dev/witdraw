package witdraw.plots

import dimwit.*
import dimwit.tensor.Labels
import dimwit.tensortree.{TensorTree, TreeOf}

import witdraw.{AxisNames, ModelResult}
import witdraw.inference.Diagnostics

/** A summary of MCMC chains, in the style of ArviZ's `summary`: one row per
  * element of each quantity, with its mean, standard deviation, highest
  * density interval, R-hat and effective sample size.
  *
  * It picks the quantities from the model's result at each draw, as a named
  * tuple, as the plots do; the fields and the axis names name the rows:
  * {{{
  * println(summary(chains, Axis[Predictor].names(predictorNames))(
  *   s => (sigma = s.sigma, beta = s.beta)
  * ))
  * // parameter          mean      sd  hdi 3%  hdi 97%   rhat     ess
  * // sigma            18.384   1.410  ...
  * // beta[male]        0.633   1.359  ...
  * }}}
  */
object Summary:

  /** One element of a quantity, over all chains. */
  type Row = (
      name: String,
      mean: Double,
      sd: Double,
      hdiLow: Double,
      hdiHigh: Double,
      rhat: Double,
      ess: Double
  )

  /** The rows of a summary, printed as a table, with the highest density
    * intervals holding a fraction `hdiProb` of the draws.
    */
  final case class Table(rows: Seq[Row], hdiProb: Double):
    override def toString: String =
      val width = (rows.map(_.name.length) :+ "parameter".length).max
      val low = f"${(1 - hdiProb) / 2 * 100}%.0f%%"
      val high = f"${(1 + hdiProb) / 2 * 100}%.0f%%"
      val header =
        s"%-${width}s %9s %9s %9s %9s %6s %7s".format("parameter", "mean", "sd", s"hdi $low", s"hdi $high", "rhat", "ess")
      val lines = rows.map: r =>
        s"%-${width}s %9.3f %9.3f %9.3f %9.3f %6.3f %7.0f".format(r.name, r.mean, r.sd, r.hdiLow, r.hdiHigh, r.rhat, r.ess)
      (header +: lines).mkString("\n")

  /** The summary of the fields of `select`, from the chains of a sampler, all
    * of the same length.
    *
    * @param axisNames
    *   names for the entries along some axes, used in the row names, e.g.
    *   `Axis[Predictor].names(predictorNames)`; by default the entries are
    *   numbered, `beta[0]`, `beta[1]`, ...
    * @param select
    *   the quantities to summarise, from the model's result at a draw
    * @param hdiProb
    *   the fraction of the draws in the highest density intervals; 94% by
    *   default, as in ArviZ
    */
  def summary[P, I, A, R](
      chains: Seq[witdraw.inference.Chain[P, I]],
      axisNames: AxisNames[?]*
  )(using ModelResult[P, A])(select: A => R, hdiProb: Double = 0.94)(using
      TensorTree[R],
      TreeOf[R, Float32],
      FieldNames[R]
  ): Table =
    require(hdiProb > 0 && hdiProb < 1, s"hdiProb must be between 0 and 1, but is $hdiProb")
    val draws = Draws.ofRecords(chains.map(_.samples.map(select)), axisNames)
    Table(draws.flatMap(rows(_, hdiProb)), hdiProb)

  /** One row per element of the quantity. R-hat needs at least 4 draws per
    * chain, and the effective sample size 2; with fewer, they are NaN.
    */
  private def rows[T <: Tuple](draws: Draws[T], hdiProb: Double): Seq[Row] =
    given Labels[T] = draws.labels
    val rhat =
      if draws.numDraws >= 4 then Draws.flat(Diagnostics.rhat(draws.chains))
      else Array.fill(draws.numElements)(Double.NaN)
    val ess =
      if draws.numDraws >= 2 then Draws.flat(Diagnostics.ess(draws.chains))
      else Array.fill(draws.numElements)(Double.NaN)
    draws.elementNames.zipWithIndex.map: (name, e) =>
      val all = draws.values.flatMap(_.map(_(e))).toArray
      val mean = all.sum / all.length
      val sd = math.sqrt(all.map(x => (x - mean) * (x - mean)).sum / all.length)
      val (low, high) = Draws.hdi(all.sorted, hdiProb)
      (name = name, mean = mean, sd = sd, hdiLow = low, hdiHigh = high, rhat = rhat(e), ess = ess(e))
