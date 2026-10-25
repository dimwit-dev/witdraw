# Quickstart: Bayesian linear regression

This guide introduces the main ideas of WitDraw through one running example,
Bayesian linear regression. By the end you will have written a model, drawn
samples from it, and fitted it to data with the No-U-Turn Sampler (NUTS).

## The big picture

A WitDraw program has four ingredients:

- **Sites** name the random variables of a model and fix their tensor types.
- A **model** is an ordinary Scala method that describes how data is
  generated. It needs a `ProbContext` to run, which it receives as a
  `using` parameter.
- **Interpreters** provide that context and so give the model its meaning.
  `ForwardSampler` runs it and draws a value for every random variable;
  `ModelScorer` computes its log-density for given values.
- **Handlers** sit between a model and the interpreter and change what the
  model's random variables do. `observe` is a handler: it fixes some sites to
  known values.

Everything else - prior predictive checks, posterior inference - is built from
these pieces.

## The model

We want to model outcomes $y_n$ as a linear function of features $x_n$, plus
Gaussian noise:

$$
\begin{aligned}
\alpha &\sim \mathcal{N}(0, 10^2) && \text{intercept} \\
\beta  &\sim \mathcal{N}(0, 10^2 I) && \text{weights, one per feature} \\
\sigma &\sim \text{HalfCauchy}(0, 5) && \text{noise scale} \\
y_n &\sim \mathcal{N}(\alpha + x_n \cdot \beta,\ \sigma^2) && n = 1, \dots, N
\end{aligned}
$$

## Setup

We import [dimwit](https://github.com/dimwit-dev/dimwit), which provides the
tensors and distributions, and WitDraw itself:

```scala mdoc
import dimwit.*
import dimwit.stats.*
import witdraw.*
import witdraw.inference.{Diagnostics, NUTS}
```

Every program that uses dimwit must initialise the dimwit runtime first:

```scala mdoc:silent
dimwit.initialize()
```

## Sites

dimwit tensors have *labelled* axes, so we first define a label for each kind
of axis: one entry per feature and one per sample (observation).

Then we declare one site per random variable. A site is an `object`, so it
has a name that we can refer to later, for example when we condition on
observed data. Its type parameter is the tensor type of the variable.

```scala mdoc
trait Feature derives Label
trait Sample derives Label

object Alpha extends Site[Tensor0[Float32]]         // intercept
object Beta extends Site[Tensor1[Feature, Float32]] // regression weights
object Sigma extends Site[Tensor0[Float32]]         // noise scale (> 0)
object Y extends Site[Tensor1[Sample, Float32]]     // outcomes
```

## Writing the model

A model is a method with a `(using ProbContext)` parameter. The line

```scala
val beta = Beta ~ Normal.isotropic(...)
```

reads "the site `Beta` is distributed according to this normal distribution".
It gives back a plain tensor of the site's type (`Tensor1[Feature, Float32]`),
which we can use in ordinary dimwit code. *Which* tensor depends on the
context the model runs in: a sampler draws one, a scorer uses a value we
supply.

The model takes the feature matrix `x` as an argument: it uses the covariates
both to compute the mean of `y` and to find out how many weights there are.
The outcomes `y`, on the other hand, are a random variable of the model. We
pass in the observed values later, by conditioning.

```scala mdoc
def model(x: Tensor2[Sample, Feature, Float32])(using ProbContext) =
  val alpha = Alpha ~ Normal(0f, 10f)
  val beta = Beta ~ Normal.isotropic(
    Tensor1(x.extent(Axis[Feature])).fill(0f),
    10f
  )
  val sigma = Sigma ~ HalfCauchy(0f, 5f)
  val mu = x.dot(Axis[Feature])(beta) +! alpha
  val y = Y ~ Normal.isotropic(mu, sigma)
  (alpha = alpha, beta = beta, sigma = sigma, y = y)
```

The model returns a named tuple of all its variables. This is what
`ForwardSampler` gives back, so naming the fields makes the samples easy to
use.

## Synthetic data

To check that inference works, we simulate data where we know the answer.
dimwit uses explicit random keys (as in JAX). We split one root key into
independent keys, one for each use, so that no key is used twice:

```scala mdoc:silent
val (dataKey, noiseKey, priorKey, fixedKey, nutsKey) =
  val ks = Key(42).split(5)
  (ks(0), ks(1), ks(2), ks(3), ks(4))

val sampleExtent = Axis[Sample] -> 100
val featureExtent = Axis[Feature] -> 2

// independent standard-normal features
val xData = Normal.standardNormal(Shape(sampleExtent, featureExtent)).sample(dataKey)

// y = alpha + x · beta + sigma * eps
val alphaTrue = Tensor0(0.5f)
val betaTrue = Tensor(Shape(featureExtent)).fromArray(Array(-1.5f, 2.0f))
val sigmaTrue = Tensor0(0.8f)
val eps = Normal.standardNormal(Shape(sampleExtent)).sample(noiseKey)
val yData = (xData.dot(Axis[Feature])(betaTrue) +! alphaTrue) + (eps *! sigmaTrue)
```

The features are drawn independently on purpose. If one feature were a
multiple of the other, only a combination of the weights could be learned
from the data, and no inference algorithm could recover `betaTrue`.

## Sampling from the model

A model can only run inside a context, so to draw from it we hand
`model(xData)` to `ForwardSampler`. (Calling `model(xData)` on its own, outside
an interpreter, is a compile error.) With no site conditioned, this samples
every variable from its prior, so the `y` values are draws from the *prior
predictive* distribution:

```scala mdoc:silent
val priorPredictive = priorKey.split(5).map(k => ForwardSampler.run(k)(model(xData)))
```

Each sample is a named tuple, so we can print a short summary of each:

```scala mdoc
priorPredictive.foreach { s =>
  println(f"alpha=${s.alpha.item}%7.2f  sigma=${s.sigma.item}%7.2f  mean(y)=${s.y.mean.item}%8.2f")
}
```

The spread of these draws shows how vague our priors are.

The handler `observe` runs a block of code with some sites fixed to given
values. Here we fix the noise scale to 1 and let everything else vary:

```scala mdoc:silent
val withFixedSigma = fixedKey.split(5).map { k =>
  ForwardSampler.run(k):
    observe(Sigma := Tensor0(1f)):
      model(xData)
}
```

```scala mdoc
withFixedSigma.foreach { s =>
  println(f"sigma=${s.sigma.item}%4.2f  mean(y)=${s.y.mean.item}%8.2f")
}
```

## Posterior inference

Now we condition on the observed outcomes and ask which parameter values are
plausible. The model observed on the data has three *latent* sites left,
`Alpha`, `Beta` and `Sigma`: the parameters whose posterior we want.

The inference methods in WitDraw work on a `Latent`: the latent sites, a
parameter type for them, and the posterior's log-density as a function of
that type. `Latent.discover` builds it. It runs the observed model forward
once, records every site with the `withTrace` handler, and collects the sites
that are not observed:

```scala mdoc:silent
val latent = Latent.discover(Constraint(Sigma, Bijector.exp)):
  observe(Y := yData):
    model(xData)
```

```scala mdoc
latent.sites.map(_.name)
```

`latent.Params` is a parameter type made for this model: it holds one value
per latent site, and `params(Sigma)` reads the value of a site, typed by the
site (here a `Sigma.Value`). `latent.logDensity` is the log-density as a
function of `latent.Params`, and `latent.init` is a starting point for
inference: the values of the traced run.

Handlers nest like any other block, and each one sees every random variable
that is sampled inside its block. To compute `logDensity`, `Latent` wraps the
model in one more `observe` that fixes the latent sites to the given
parameters, so `ModelScorer` finds every site observed and adds up their
log-densities.

### Handling the constraint on sigma

NUTS moves freely through all of $\mathbb{R}^n$, but $\sigma$ must be positive.
If we sampled $\sigma$ directly, the sampler would sooner or later try
negative values, where the model's density is undefined. The standard fix is
to sample $z = \log \sigma$ instead, which can be any real number.

A `Bijector` describes such a change of variables, and
`Constraint(Sigma, Bijector.exp)` above applies one to `Sigma`. The sampler
then moves in unconstrained space, and `latent.logDensity` includes the
Jacobian correction for the change of variables. We never see $z$ ourselves:
`params(Sigma)` always gives $\sigma = e^z$.

### Running NUTS

NUTS is a variant of Hamiltonian Monte Carlo that chooses the length of each
trajectory automatically. It still has two tuning parameters: the step size,
and the mass matrix, which rescales each direction of the parameter space and
works best when it approximates the posterior covariance. `NUTS.adapt` runs a
warmup that tunes both (Stan's window adaptation, run by BlackJax). The warmup
also moves the chain away from its starting point, so no draws need to be
discarded afterwards: we start the tuned sampler where the warmup ended.

```scala mdoc:silent
val (warmupKey, sampleKey) = nutsKey.split2()
val adapted = NUTS.adapt(latent, latent.init, warmupKey, numSteps = 1000)
val nSamples = 2000
val (samples, infos) = adapted.sampler
  .run(adapted.position, sampleKey)
  .take(nSamples)
  .toList
  .unzip
```

The tuned step size is `adapted.stepSize`. By default the mass matrix is
diagonal, and `adapted.inverseMassMatrix` holds the estimated posterior
variance of each parameter; `dense = true` estimates the full covariance
instead, at the cost of a longer warmup. To choose the parameters yourself,
use `NUTS(latent, stepSize = 0.05)`, optionally with an `InverseMassMatrix`.

### Checking the result

Finally we stack the draws along a new axis and compare the posterior mean
and standard deviation of each parameter with the true values:

```scala mdoc:silent
trait Draw derives Label

def summary[T <: Tuple: Labels](draws: List[Tensor[T, Float32]]) =
  val stacked = stack(draws, Axis[Draw])
  (stacked.mean(Axis[Draw]), stacked.std(Axis[Draw]))

val (alphaMean, alphaSd) = summary(samples.map(_(Alpha)))
val (betaMean, betaSd) = summary(samples.map(_(Beta)))
val (sigmaMean, sigmaSd) = summary(samples.map(_(Sigma)))
```

```scala mdoc
println(s"alpha: mean $alphaMean, sd $alphaSd   (true $alphaTrue)")
println(s"beta:  mean $betaMean, sd $betaSd   (true $betaTrue)")
println(s"sigma: mean $sigmaMean, sd $sigmaSd   (true $sigmaTrue)")
```

The true values lie within roughly two posterior standard deviations of the
posterior means. With only 100 noisy observations we cannot expect an exact
match.

Each draw also comes with diagnostic information from the sampler. A healthy
run has no divergent transitions and a reasonably high acceptance rate:

```scala mdoc
val divergences = infos.count(_.isDivergent)
val meanAcceptance = infos.map(_.acceptanceRate).sum / infos.size
```

### Running several chains

One chain cannot tell us whether it has explored the whole posterior. The
standard check is to run several chains from different starting points and
compare them. `latent.initRandom` draws a starting point (uniformly between -2
and 2 in unconstrained space, as Stan does), and `NUTS.adaptChains` tunes each
chain on its own. All chains then run side by side, in one compiled loop:

```scala mdoc:silent
val Seq(initKey, chainsWarmupKey, chainsSampleKey) = nutsKey.split(3)
val starts = initKey.split(4).map(latent.initRandom(_))
val adaptedChains = NUTS.adaptChains(latent, starts, chainsWarmupKey)
val chains = adaptedChains.sample(chainsSampleKey, 1000).map(_.map(_._1))
```

`Diagnostics.rhat` compares the chains: values close to 1 mean they agree, and
values above about 1.01 are a warning. `Diagnostics.ess` estimates how many
independent draws the chains are worth. Both have the shape of the quantity,
so `beta` gets one value per predictor:

```scala mdoc
Diagnostics.rhat(chains.map(_.map(_(Beta))))
Diagnostics.ess(chains.map(_.map(_(Sigma))))
```

For a sampler with fixed settings, `sampler.sampleChains(starts, key, numDraws)`
does the same without the warmup.

NUTS is not the only inference method: `MAP(latent)` finds the posterior mode,
and `Laplace(latent)` approximates the posterior by a Gaussian around its mode
and is run like NUTS. Both are in `witdraw.inference` and take the same
`latent`.

## Composing models

`model` above tells the whole story in one method. As models grow, it helps
to split them into parts, for example to reuse a likelihood with different
priors. Since a model is an ordinary method, this is ordinary Scala: a part of
a model is a method with a `(using ProbContext)` parameter, and calling it
from another model runs it in the same context.

The prior is a part of its own. It returns the parameters as a named tuple:

```scala mdoc:silent
type Params = (alpha: Alpha.Value, beta: Beta.Value, sigma: Sigma.Value)

def prior(nFeatures: Int)(using ProbContext): Params =
  val alpha = Alpha ~ Normal(0f, 10f)
  val beta = Beta ~ Normal.isotropic(
    Tensor1(Axis[Feature] -> nFeatures).fill(0f),
    10f
  )
  val sigma = Sigma ~ HalfCauchy(0f, 5f)
  (alpha = alpha, beta = beta, sigma = sigma)
```

The likelihood takes the parameters as ordinary tensors. It does not know
where they come from, only that they have the right types:

```scala mdoc:silent
def likelihood(
    x: Tensor2[Sample, Feature, Float32],
    params: Params
)(using ProbContext) =
  val mu = x.dot(Axis[Feature])(params.beta) +! params.alpha
  Y ~ Normal.isotropic(mu, params.sigma)
```

The full model takes the prior as a parameter, so that we can swap it. A
model passed around as a value has the type `ProbProg[A]`, short for the
context function type `ProbContext ?=> A`. Writing `prior` inside the model
runs it, in the model's own context:

```scala mdoc:silent
def regression(
    x: Tensor2[Sample, Feature, Float32],
    prior: ProbProg[Params]
)(using ProbContext) =
  val params = prior
  val y = likelihood(x, params)
  (alpha = params.alpha, beta = params.beta, sigma = params.sigma, y = y)
```

`regression(xData, prior(nFeatures))` contains the same sites with the same
distributions as `model(xData)`, so it can be used in exactly the same way,
from `ForwardSampler` to `Latent.discover` and NUTS. For example, both give
the same log-density at the true parameter values:

```scala mdoc
val nFeatures = xData.shape(Axis[Feature])

def logDensityAtTruth(m: ProbProg[Any]): Float =
  ModelScorer.run:
    observe(Y := yData, Alpha := alphaTrue, Beta := betaTrue, Sigma := sigmaTrue):
      m
  .item

(logDensityAtTruth(model(xData)), logDensityAtTruth(regression(xData, prior(nFeatures))))
```

Because the prior is a parameter of `regression`, trying a different prior
does not touch the likelihood. Any program that yields `Params` fits,
for example one with tighter priors on the weights:

```scala mdoc:silent
def tightPrior(nFeatures: Int)(using ProbContext): Params =
  val alpha = Alpha ~ Normal(0f, 10f)
  val beta = Beta ~ Normal.isotropic(
    Tensor1(Axis[Feature] -> nFeatures).fill(0f),
    1f
  )
  val sigma = Sigma ~ HalfCauchy(0f, 5f)
  (alpha = alpha, beta = beta, sigma = sigma)

val tightSamples = priorKey.split(5).map { k =>
  ForwardSampler.run(k)(regression(xData, tightPrior(nFeatures)))
}
```

Two things to keep in mind when composing:

- A part that samples sites needs a `ProbContext`, so it can only be called
  from a model, a handler or an interpreter. Calling it anywhere else is a
  compile error.
- A site is identified by its object: two different objects are different
  sites, even if they share a name. Each site may be sampled only once per
  run when it is observed, traced or inferred; there, sampling it twice is an
  error.

## Components

Methods are enough to split a model into parts, but each part still uses
the global sites `Alpha`, `Beta` and `Sigma`. A part that samples sites can
therefore be used only once per model. Calling `likelihood` twice, say for
two outcomes, would sample `Y` twice.

A *component* is a part of a model that owns its sites. It is written as a
module: a trait that extends `Component` declares the part, and each object
that extends the trait is one instance of it, with its own sites.

As an example we let the noise scale depend on the features too, a
*location-scale* regression:

$$
y_n \sim \mathcal{N}\big(a + x_n \cdot b,\ \exp(c + x_n \cdot d)^2\big)
$$

Both the mean and the log-scale are linear in the features, so we write the
linear predictor once, as a component. Its prior scale is a trait parameter:

```scala mdoc:silent
trait Linear(priorScale: Float) extends Component:
  object Intercept extends Site[Tensor0[Float32]]
  object Weights extends Site[Tensor1[Feature, Float32]]

  def apply(x: Tensor2[Sample, Feature, Float32])(using ProbContext) =
    val intercept = Intercept ~ Normal(0f, priorScale)
    val weights = Weights ~ Normal.isotropic(
      Tensor1(x.extent(Axis[Feature])).fill(0f),
      priorScale
    )
    x.dot(Axis[Feature])(weights) +! intercept

  def values(v: SiteValues) = Linear.Values(v(Intercept), v(Weights))

object Linear:
  case class Values(intercept: Tensor0[Float32], weights: Tensor1[Feature, Float32])
```

The method `values` and its record `Linear.Values` are optional; we come
back to them below.

A component can contain other components. The location-scale regression
has two instances of `Linear`, and its own outcome site:

```scala mdoc:silent
trait LocationScale extends Component:
  object mean extends Linear(priorScale = 10f)
  object logScale extends Linear(priorScale = 1f)
  object Y extends Site[Tensor1[Sample, Float32]]

  def apply(x: Tensor2[Sample, Feature, Float32])(using ProbContext) =
    Y ~ Normal(mean(x), logScale(x).exp)

  def values(v: SiteValues) = LocationScale.Values(mean.values(v), logScale.values(v))

object LocationScale:
  case class Values(mean: Linear.Values, logScale: Linear.Values)

object outcome extends LocationScale
```

A site is named by the path of the components it is declared in. This is
how sites with the same declaration are told apart, in error messages as
well as here:

```scala mdoc
val lsLatent = Latent.discover():
  observe(outcome.Y := yData):
    outcome(xData)

lsLatent.sites.map(_.name)
```

No constraint is needed: the scale is positive through `exp`, so all
parameters can take any real value.

### Values as a typed record

`params(outcome.mean.Weights)` reads one value, but with many sites it helps
to read all values of a component at once, as a typed record. This is what
the `values` methods above do, and the records nest like the components do:
`LocationScale.Values` holds one `Linear.Values` for the mean and one for
the log-scale.

A `SiteValues` gives the value of any site. Both a `Trace` from `withTrace`
and `lsLatent.values(params)` are one, so the same method reads a record
from a run of the model and from a point of the posterior. Here we find the
posterior mode with `MAP` and read it as a record. We start the search at 0,
i.e. mean 0 and scale 1 for every sample, rather than at the traced values,
a draw from the prior whose scales can be extreme:

```scala mdoc:silent
import witdraw.inference.MAP

val zeros = Tensor1(xData.extent(Axis[Feature])).fill(0f)
val start = Seq(
  outcome.mean.Intercept := Tensor0(0f),
  outcome.mean.Weights := zeros,
  outcome.logScale.Intercept := Tensor0(0f),
  outcome.logScale.Weights := zeros
)
```

```scala mdoc
val mode = outcome.values(lsLatent.values(MAP(lsLatent, start)))
mode.mean.weights
mode.logScale.weights
mode.logScale.intercept.exp
```

The weights of the mean are close to `betaTrue`. The data was simulated
with a constant noise scale, so the weights of the log-scale are close to 0,
and the scale itself, $e^c$, is close to the noise scale $\sigma$ that NUTS
found for the first model.

Records of tensors are DimWit tensor trees, so draws can also be averaged,
mapped over or saved as a whole. The example
[LocationScale.scala](../examples/src/main/scala/witdraw/examples/components/LocationScale.scala)
fits a model with two outcomes, each a `LocationScale`, with NUTS, and
averages the posterior draws as records.

A component that is not an object, e.g. one of several created in a loop,
needs an explicit name: `new Linear(1f) { override def name = s"region$i" }`.
