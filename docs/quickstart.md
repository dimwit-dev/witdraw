# Quickstart: Bayesian linear regression

This guide introduces the main ideas of WitDraw through one running example,
Bayesian linear regression. By the end you will have written a model, drawn
samples from it, and fitted it to data with the No-U-Turn Sampler (NUTS).

## The big picture

A WitDraw program has three ingredients:

- **Sites** name the random variables of a model and fix their tensor types.
- A **model** (`ProbProg`) is a *description* of how data is generated. Writing
  it down does not draw any random numbers.
- **Interpreters** give a model meaning. `ForwardSampler` runs it and draws a
  value for every random variable; `ModelScorer` computes its log-density for
  given values. Conditioning (`observe`) fixes some sites to known values
  before an interpreter runs the model.

Everything else - prior predictive checks, posterior inference - is built from
these three pieces.

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

```scala
import dimwit.*
import dimwit.stats.*
import witdraw.*
import witdraw.mcmc.NUTSAlgorithm
import witdraw.reparam.{Bijector, Reparam}
```

Every program that uses dimwit must initialise the dimwit runtime first:

```scala
dimwit.initialize()
```

## Sites

dimwit tensors have *labelled* axes, so we first define a label for each kind
of axis: one entry per feature and one per sample (observation).

Then we declare one site per random variable. A site is an `object`, so it
has a name that we can refer to later, for example when we condition on
observed data. Its type parameter is the tensor type of the variable.

```scala
trait Feature derives Label
trait Sample derives Label

object Alpha extends Site[Tensor0[Float32]]         // intercept
object Beta extends Site[Tensor1[Feature, Float32]] // regression weights
object Sigma extends Site[Tensor0[Float32]]         // noise scale (> 0)
object Y extends Site[Tensor1[Sample, Float32]]     // outcomes
```

## Writing the model

A model is written inside a `ProbProg:` block. The line

```scala
val beta = Beta ~ Normal.isotropic(...)
```

reads "the site `Beta` is distributed according to this normal distribution".
It gives back a plain tensor of the site's type (`Tensor1[Feature, Float32]`),
which we can use in ordinary dimwit code. *Which* tensor depends on the
interpreter: a sampler draws one, a scorer uses a value we supply.

The model takes the feature matrix `x` as an argument: it uses the covariates
both to compute the mean of `y` and to find out how many weights there are.
The outcomes `y`, on the other hand, are a random variable of the model. We
pass in the observed values later, by conditioning.

```scala
def model(x: Tensor2[Sample, Feature, Float32]) = ProbProg:
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

```scala
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

`model(xData)` is still only a description. To draw from it, we run it with
`ForwardSampler`. With no site conditioned, this samples every variable from
its prior, so the `y` values are draws from the *prior predictive*
distribution:

```scala
val modelOnX = model(xData)

val priorPredictive = priorKey.split(5).map(k => ForwardSampler.run(k)(modelOnX))
```

Each sample is a named tuple, so we can print a short summary of each:

```scala
priorPredictive.foreach { s =>
  println(f"alpha=${s.alpha.item}%7.2f  sigma=${s.sigma.item}%7.2f  mean(y)=${s.y.mean.item}%8.2f")
}
// alpha=   3.57  sigma=   2.96  mean(y)=    3.51
// alpha=   0.02  sigma=   3.84  mean(y)=   -1.06
// alpha=  23.41  sigma=   1.85  mean(y)=   22.30
// alpha=   0.68  sigma=  53.42  mean(y)=    1.33
// alpha= -10.14  sigma=  13.13  mean(y)=  -10.02
```

The spread of these draws shows how vague our priors are.

`observe` fixes sites to given values. Here we fix the noise scale to 1 and
let everything else vary. `Sigma` comes *before* the other variables in the
model and nothing depends on it except `y`, so in a forward sampler this
simply sets its value; the other draws are unaffected:

```scala
val withFixedSigma = fixedKey.split(5).map { k =>
  ForwardSampler.run(k)(modelOnX.observe(Sigma := Tensor0(1f)))
}
```

```scala
withFixedSigma.foreach { s =>
  println(f"sigma=${s.sigma.item}%4.2f  mean(y)=${s.y.mean.item}%8.2f")
}
// sigma=1.00  mean(y)=   -2.25
// sigma=1.00  mean(y)=   11.51
// sigma=1.00  mean(y)=    8.88
// sigma=1.00  mean(y)=    2.48
// sigma=1.00  mean(y)=  -16.95
```

## Posterior inference

Now we condition on the observed outcomes and ask which parameter values are
plausible. Most inference algorithms in WitDraw need the (unnormalised) log
density of the posterior as a function of the parameters. We get it by
conditioning the model on both the data and a candidate parameter value, and
scoring it with `ModelScorer`:

```scala
type PosteriorParams =
  (alpha: Alpha.Value, beta: Beta.Value, sigma: Sigma.Value)

val logDensity = (params: PosteriorParams) =>
  ModelScorer.run(modelOnX.observe(Y := yData).observeAll(params))
```

`observeAll` matches the fields of the named tuple to sites **by name**: a
field `sigma` conditions the site `object Sigma` (the first letter of the site
name lower-cased). A field whose name matches no site is ignored, so check the
spelling.

### Handling the constraint on sigma

NUTS moves freely through all of $\mathbb{R}^n$, but $\sigma$ must be positive.
If we sampled $\sigma$ directly, the sampler would sooner or later try
negative values, where the model's density is undefined. The standard fix is
to sample $z = \log \sigma$ instead, which can be any real number.

A `Bijector` describes such a change of variables, and `Reparam.build`
combines one bijector per field into a bijector for the whole parameter
tuple; fields without one, here `alpha` and `beta`, are left unchanged.
`Reparam.pushForward` then turns the log-density over the original
parameters into one over the unconstrained parameters, including the Jacobian
correction:

```scala
val toConstrained = Reparam.build[PosteriorParams, PosteriorParams](
  (sigma = Bijector.exp[EmptyTuple])
)
val unconstrainedLogDensity = Reparam.pushForward(toConstrained)(logDensity)
```

### Running NUTS

NUTS is a variant of Hamiltonian Monte Carlo that chooses the length of each
trajectory automatically. We still have to choose a step size: the sampler
does not adapt it, so if you see many divergences or a low acceptance rate,
reduce it.

The sampler runs in unconstrained space, so we map the initial value there
with `inverse`, and map the draws back with `forward`. We discard the first
draws (burn-in) while the chain moves away from its starting point.

```scala
val initParams: PosteriorParams = (
  alpha = Tensor0(0f),
  beta = Tensor1(featureExtent).fill(0f),
  sigma = Tensor0(1f)
)

val burnIn = 500
val nSamples = 2000
val nuts = NUTSAlgorithm[PosteriorParams](stepSize = 0.05)(unconstrainedLogDensity)
val (zSamples, infos) = nuts
  .run(toConstrained.inverse(initParams), nutsKey)
  .drop(burnIn)
  .take(nSamples)
  .toList
  .unzip
val samples = zSamples.map(toConstrained.forward)
```

### Checking the result

Finally we stack the draws along a new axis and compare the posterior mean
and standard deviation of each parameter with the true values:

```scala
trait Draw derives Label

def summary[T <: Tuple: Labels](draws: List[Tensor[T, Float32]]) =
  val stacked = stack(draws, Axis[Draw])
  (stacked.mean(Axis[Draw]), stacked.std(Axis[Draw]))

val (alphaMean, alphaSd) = summary(samples.map(_.alpha))
val (betaMean, betaSd) = summary(samples.map(_.beta))
val (sigmaMean, sigmaSd) = summary(samples.map(_.sigma))
```

```scala
println(s"alpha: mean $alphaMean, sd $alphaSd   (true $alphaTrue)")
// alpha: mean 0.40087232, sd 0.069597654   (true 0.5)
println(s"beta:  mean $betaMean, sd $betaSd   (true $betaTrue)")
// beta:  mean [-1.4122269  2.0413063], sd [0.07261302 0.06949434]   (true [-1.5  2. ])
println(s"sigma: mean $sigmaMean, sd $sigmaSd   (true $sigmaTrue)")
// sigma: mean 0.69023913, sd 0.05033151   (true 0.8)
```

The true values lie within about two posterior standard deviations of the
posterior means. With only 100 noisy observations we cannot expect an exact
match.

Each draw also comes with diagnostic information from the sampler. A healthy
run has no divergent transitions and a reasonably high acceptance rate:

```scala
val divergences = infos.count(_.isDivergent)
// divergences: Int = 0
val meanAcceptance = infos.map(_.acceptanceRate).sum / infos.size
// meanAcceptance: Double = 0.9209054022133351
```
