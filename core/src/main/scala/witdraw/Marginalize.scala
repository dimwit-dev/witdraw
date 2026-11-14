package witdraw

/** Runs `model` with the given discrete sites marked to be summed out by
  * inference. The model stays as it is, and so does its distribution: a
  * forward run, e.g. with [[ForwardSampler]], still draws these sites.
  *
  * [[Latent.discover]] sums them out: its parameters are those of the other
  * latent sites, and its log-density is the marginal one, so that NUTS, MAP or
  * Laplace work on the posterior `p(theta | x)` of those other sites. The sites
  * summed out come back with [[Latent.recover]], which draws them from their
  * exact conditional posterior `p(z | theta, x)` at every draw of a chain, or
  * with [[Latent.posteriorOf]], which gives that posterior itself.
  * {{{
  * val posterior = Latent.discover:
  *   marginalize(SwitchPoint):
  *     observe(Disasters := counts)(model(years))
  * }}}
  *
  * A site summed out must be scalar, latent, and have finitely many values,
  * which must not depend on the other sites: a `DiscreteUniform` with fixed
  * bounds, a `Categorical` or a `Bernoulli`. Every log-density evaluation runs
  * the model once for every value, or for every combination of values of
  * several sites.
  *
  * The mark only asks inference to sum the site out; the model can still be
  * conditioned on the site from outside, e.g. a component that sums out its
  * own site, used with that site observed. An observation outside the block
  * wins: the site is then observed, not summed out.
  * {{{
  * observe(SwitchPoint := Tensor0(40)):
  *   marginalize(SwitchPoint)(model)     // SwitchPoint is observed
  * }}}
  *
  * It is an error to give a site twice, a site the model never samples, or a
  * site observed inside the block, which contradicts summing it out.
  */
def marginalize[A](sites: SiteDef[?, ?]*)(model: ProbProg[A])(using
    ProbContext
): A =
  intercept("marginalized", "marginalize", sites.map(_ -> ()))(
    [T <: Tuple, V] =>
      (statement: SampleStatement[T, V], _: Unit) =>
        if statement.observed.isDefined then
          throw IllegalArgumentException(
            s"Site '${statement.site.name}' is marginalized, but it is observed inside the marginalize block. " +
              "Only latent sites can be summed out; to fix its value, observe it outside the block."
          )
        statement.copy(marginalized = true)
  )(model)
