# Quickstart: Bayesian linear regression

In WitDraw, a model is a joint probability distribution over all quantities
of a problem: the parameters we want to learn and the data we observe. Once
the joint distribution is written down, a Bayesian analysis consists of asking
it questions: drawing from it, evaluating its density and, above all,
conditioning it on the data we have observed. This guide develops these ideas
on one running example, Bayesian linear regression.

## A model is a joint distribution

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

These four lines define one joint distribution over all random variables of
the problem, given the features $x = (x_1, \dots, x_N)$:

$$
p(\alpha, \beta, \sigma, y \mid x)
  = p(\alpha)\, p(\beta)\, p(\sigma) \prod_{n=1}^N p(y_n \mid \alpha, \beta, \sigma, x_n).
$$

Each line contributes the distribution of its variable given the variables it
depends on: the first three lines one factor each, the last line one factor
per observation. The model does not describe how the features arise: they are
fixed inputs, which is why they appear only to the right of the bar.

The same factorisation can be read as a recipe for generating data: draw
$\alpha$, $\beta$ and $\sigma$ from their priors, then each $y_n$ given them.
This is why such a model is called *generative*. The recipe and the joint
distribution are one and the same thing, seen in two ways: the order in which
we draw the variables is the order of the factors.

Every task in this guide is a question asked of this one joint distribution.
The two most basic questions are: what do draws from it look like, and what is
its density at given values of all variables? The question that a data
analysis is about, though, starts from the outcomes $y_\text{obs}$ we have
actually observed. Fixing $y$ to $y_\text{obs}$ and asking what this tells us
about the remaining variables is *conditioning*, and its answer is the
*posterior* distribution $p(\alpha, \beta, \sigma \mid y_\text{obs}, x)$. As
we will see, the posterior needs nothing beyond the joint distribution: up to a
constant, its density is the joint density with $y$ fixed to the data.

A WitDraw model is a direct translation of the factorisation: each line above
becomes one statement in the program. Each concept so far has its counterpart
in WitDraw:

| Concept | In WitDraw |
| --- | --- |
| A random variable | a `Site` |
| One factor $p(v \mid \dots)$ | a statement `Site ~ distribution` |
| The joint distribution | a model: a method that runs these statements |
| Drawing from the joint | `ForwardSampler` |
| Evaluating the joint density | `ModelScorer` |
| Fixing variables to given values, e.g. the data | the handler `observe` |
| The posterior: the density of the other variables, up to a constant | `Latent` |

An *interpreter*, such as `ForwardSampler` or `ModelScorer`, does not change
what the model means: the model always describes the same joint distribution.
The interpreter decides which computation is performed with it. A *handler*,
such as `observe`, sits between the model and the interpreter and changes
what individual random variables do, here by fixing their values.

## Setup

We import [dimwit](https://github.com/dimwit-dev/dimwit), which provides the
tensors and distributions, and WitDraw itself:

```scala mdoc
import dimwit.*
import dimwit.stats.*
import witdraw.*
import witdraw.inference.NUTS
import witdraw.inference.Summary.summary
```

Every program that uses dimwit must initialise the dimwit runtime first:

```scala mdoc:silent
dimwit.initialize()
```

## Sites

dimwit tensors have *labelled* axes, so we first define a label for each kind
of axis: one entry per feature and one per sample (observation).

Then we declare one site per random variable: `Alpha`, `Beta`, `Sigma` and `Y`
stand for $\alpha$, $\beta$, $\sigma$ and $y = (y_1, \dots, y_N)$. A site is an
`object`, so it has a name that we can refer to later, for example when we
condition on observed data. Its type parameter is the tensor type of the
variable: $\beta$ is a vector with one entry per feature, $y$ one with one
entry per sample.

A site stands for the variable, not for a value of it. In the code below, the
features $x$ are `xData`, and the observed outcomes $y_\text{obs}$ are
`yData`, a value we will give to the site `Y`.

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

Line by line, this is the model from the beginning: one `~` statement per
line, each with the distribution of its variable given the values computed
before it. The statements for $\alpha$, $\beta$ and $\sigma$ are one factor
each. The statement for `Y` is all $N$ factors $\prod_n p(y_n \mid \alpha, \beta, \sigma, x_n)$
at once: `y` is a vector with one entry per sample, and
`Normal.isotropic` is the distribution of $N$ independent normal variables, so
its density is the product of their densities. Two details differ from the
mathematical notation: dimwit parametrises a normal distribution by its
standard deviation, not its variance, so $\mathcal{N}(0, 10^2)$ becomes
`Normal(0f, 10f)`; and the mean `mu` is not a site, because it is not random:
values computed from other variables are ordinary code.

The order of the statements is the order of the factors. The model's density
is the product of the factors, so its log-density is the sum of the
log-densities of its `~` statements. This is what `ModelScorer`
computes: given an assignment of values to all variables, it runs the model
with those values and returns the log-density of the joint distribution at
them, $\log p(\alpha, \beta, \sigma, y \mid x)$. It works for any assignment,
so `ModelScorer` lets us evaluate the joint density wherever we like. It
returns the logarithm rather than the density itself, because the density is
a product of many factors, here 103, which quickly becomes too small to
represent as a floating-point number; the sum of their logarithms does not.

The model returns a named tuple of all its variables. This is what
`ForwardSampler` gives back, so naming the fields makes the samples easy to
use.

## Features

The model needs features to run: they are its inputs. We draw 100 samples of
two features at random.

dimwit uses explicit random keys (as in JAX). We split one root key into
independent keys, one for each use in this guide, so that no key is used
twice:

```scala mdoc:silent
val (featureKey, priorKey, fixedKey, dataKey, nutsKey, replicateKey) =
  val ks = Key(42).split(6)
  (ks(0), ks(1), ks(2), ks(3), ks(4), ks(5))

val sampleExtent = Axis[Sample] -> 100
val featureExtent = Axis[Feature] -> 2

// independent standard-normal features
val xData = Normal.standardNormal(Shape(sampleExtent, featureExtent)).sample(featureKey)
```

The features are drawn independently on purpose. If one feature were a
multiple of the other, only a combination of the weights could be learned
from the data, and no inference algorithm could recover the weights.

## Drawing from the joint distribution

A model can only run inside a context, so to draw from it we hand
`model(xData)` to `ForwardSampler`. (Calling `model(xData)` on its own, outside
an interpreter, is a compile error.) `ForwardSampler` follows the generative
recipe: it draws each variable in turn, given the values drawn before it. Each
run is therefore one draw $(\alpha, \beta, \sigma, y)$ from the joint
distribution.

To study only some of the variables, we ignore the others in each draw. The
`y` values on their own are then draws from the *prior predictive*
distribution $p(y \mid x)$: what the model expects data to look like before
seeing any. (Ignoring variables in draws from a joint distribution gives draws
from the marginal distribution of the rest.)

```scala mdoc:silent
val priorPredictive = ForwardSampler.sample(priorKey, 5)(model(xData))
```

`ForwardSampler.sample` runs the model 5 times, each with its own key split
from `priorKey`; `ForwardSampler.run(key)(model)` runs it once.

Each sample is a named tuple, so we can print a short summary of each:

```scala mdoc
priorPredictive.foreach { s =>
  println(f"alpha=${s.alpha.item}%7.2f  sigma=${s.sigma.item}%7.2f  mean(y)=${s.y.mean.item}%8.2f")
}
```

The spread of these draws shows how vague our priors are.

### Fixing a variable

The handler `observe` runs a block of code with some sites fixed to given
values. Here we fix the noise scale to 1 and let everything else vary:

```scala mdoc:silent
val withFixedSigma =
  ForwardSampler.sample(fixedKey, 5):
    observe(Sigma := Tensor0(1f)):
      model(xData)
```

```scala mdoc
withFixedSigma.foreach { s =>
  println(f"sigma=${s.sigma.item}%4.2f  mean(y)=${s.y.mean.item}%8.2f")
}
```

These are draws from the conditional distribution
$p(\alpha, \beta, y \mid \sigma = 1, x)$: fixing $\sigma$ and running the
recipe is conditioning on $\sigma$. This works because $\sigma$ depends on no
other variable. Knowing it tells us nothing about $\alpha$ and $\beta$, which
the recipe draws from their priors as before, and $y$ is drawn given the
value of $\sigma$ we fixed.

The same is not true for the data, because $y$ depends on $\alpha$, $\beta$
and $\sigma$. Observing `Y` inside `ForwardSampler` would draw the parameters
from their priors, exactly as before, and merely skip drawing $y$: the recipe
runs forwards, and the information in $y$ cannot flow back to the parameters
it depends on. Conditioning on the data means running the recipe backwards,
and that is the task of inference.

So `observe` always means the same thing: this variable has this value.
Whether a computation turns that into the conditional distribution of the
other variables depends on the interpreter. `ForwardSampler` does so only when
the fixed variables depend on no free variable, as for $\sigma$ here. For any
variables, and in particular for the data, this is the job of inference,
which we turn to after making some data.

## Simulating data from known parameters

To check that inference works, we need data for which we know the answer. We
already have everything to make it: we fix all three parameters to chosen
"true" values and run the model forwards. The outcomes are then one draw from
$p(y \mid \alpha, \beta, \sigma, x)$, data that the model itself could have
produced with these parameters:

```scala mdoc:silent
val alphaTrue = Tensor0(0.5f)
val betaTrue = Tensor(Shape(featureExtent)).fromArray(Array(-1.5f, 2.0f))
val sigmaTrue = Tensor0(0.8f)

val yData =
  ForwardSampler.run(dataKey):
    observe(Alpha := alphaTrue, Beta := betaTrue, Sigma := sigmaTrue):
      model(xData)
  .y
```

From now on we pretend not to know the true values, and ask what the data
tells us about them.

## Conditioning on the data

What we want is the *posterior* distribution: the distribution of the
parameters $\theta = (\alpha, \beta, \sigma)$ given the observed outcomes
$y_\text{obs}$. By the definition of conditional probability (Bayes' rule),

$$
p(\theta \mid y_\text{obs}, x)
  = \frac{p(\theta, y_\text{obs} \mid x)}{p(y_\text{obs} \mid x)}
  = \frac{p(\theta)\, p(y_\text{obs} \mid \theta, x)}{p(y_\text{obs} \mid x)}.
$$

The numerator is the joint density, with the outcomes fixed to the
observations. By the factorisation of our model it is the product of two
parts: the *prior* $p(\theta) = p(\alpha)\, p(\beta)\, p(\sigma)$, the first
three lines of the model, and the *likelihood*
$p(y_\text{obs} \mid \theta, x) = \prod_n p(y_{\text{obs},n} \mid \alpha, \beta, \sigma, x_n)$,
the last line, which measures how well the parameters
$\theta$ explain the observed data.

### The posterior as a function of the parameters

`ModelScorer` evaluates the joint log-density for any assignment of values to
all variables. The outcomes are given, so only the parameters are left to
choose, and for every $\theta$ we can compute

$$
\theta \mapsto \log p(\theta, y_\text{obs} \mid x),
$$

a function of the parameters alone. We can write it down directly, with
`observe` fixing all sites:

```scala mdoc:silent
def logJoint(alpha: Float, beta: Array[Float], sigma: Float): Float =
  ModelScorer.run:
    observe(
      Y := yData,
      Alpha := Tensor0(alpha),
      Beta := Tensor1(Axis[Feature]).fromArray(beta),
      Sigma := Tensor0(sigma)
    ):
      model(xData)
  .item
```

At the true parameters it is far higher than at, say, $\alpha = 0$,
$\beta = 0$, $\sigma = 1$, which explain the data much worse:

```scala mdoc
logJoint(0.5f, Array(-1.5f, 2.0f), 0.8f)
logJoint(0f, Array(0f, 0f), 1f)
```

The denominator is an integral over all parameters, which we usually cannot
compute. But it does not depend on $\theta$, so the function above is the log
posterior up to a constant. That is enough: Markov chain Monte Carlo (MCMC)
methods such as NUTS draw from a distribution given only its log-density up to
a constant. The model observed on the data therefore already defines
everything inference needs.

### Finding the parameters: `Latent`

To turn the observed model into this function, we must know its arguments:
the sites that are not observed. These are the *latent* sites. `Latent.discover`
finds them: it runs the observed model forward once, records every site, and
collects those without a value:

```scala mdoc:silent
val latent = Latent.discover:
  observe(Y := yData):
    model(xData)
```

```scala mdoc
latent.sites.map(_.name)
```

`latent.logDensity` is the function above. Its argument has the type
`latent.Params`, made for this model: one value per latent site, in one
vector. To evaluate it, `Latent` fixes the latent sites to these values, so
that every site has a value, and runs `ModelScorer`.

How the values are laid out in the vector is an internal detail, so we do not
read it directly: `latent.result(params)` gives what the model returns at
given parameters, the same named tuple as the prior predictive draws above.
`latent.init` is a starting point for inference: the values of the traced
run, i.e. one draw of the parameters from the prior.

The `latent` is all an inference algorithm needs, and we hand it to NUTS next.

### Running NUTS

An MCMC method produces a sequence of draws
$\theta^{(1)}, \dots, \theta^{(S)}$, a *chain*, whose distribution approaches
the posterior. Each step moves from the current draw to a new one, using only
the log-density. Averages over the draws then estimate averages over the
posterior: the posterior mean $E[\alpha \mid y_\text{obs}, x]$, for example,
is estimated by $\frac{1}{S} \sum_s \alpha^{(s)}$.

NUTS (the No-U-Turn Sampler) is such a method. It follows the gradient of the
log-density, which lets it take large steps through the parameter space. It
has tuning parameters, which `NUTS.adapt` sets in a warmup phase. The warmup
also moves the chain away from its starting point, so no draws need to be
discarded afterwards: `adapted.sample` continues where the warmup ended, and
returns the chain; `samples` gives the model's result at each draw, the same
named tuple as the prior predictive draws above.

```scala mdoc:silent
val (warmupKey, sampleKey) = nutsKey.split2()
val adapted = NUTS.adapt(latent, warmupKey, latent.init, numSteps = 1000)
val nSamples = 2000
val nutsChain = adapted.sample(sampleKey, nSamples)
val samples = nutsChain.samples
```

### Checking the result

Finally we summarise the posterior and compare it with the true values.
`Summary.summary` takes a sequence of chains, here just one, and picks the
quantities to summarise from the model's result at each draw, as a named
tuple. It gives one row per element of each quantity, with estimates from the
draws: the posterior mean, the posterior standard deviation and the 94%
highest density interval (HDI), the narrowest interval that holds 94% of the
draws. The HDI is a *credible interval*: given the data, the parameter lies in
it with probability 0.94. The last two columns are diagnostics, which we come
back to below.

```scala mdoc
println(summary(Seq(nutsChain))(s => (alpha = s.alpha, beta = s.beta, sigma = s.sigma)))
println(s"true values: alpha $alphaTrue, beta $betaTrue, sigma $sigmaTrue")
```

All true values lie inside their 94% intervals. They do not match the
posterior means exactly: with only 100 noisy observations, the data leaves
some uncertainty about the parameters, and the standard deviations and
intervals tell us how much.

Each draw also comes with diagnostic information from the sampler. A healthy
run has no divergent transitions and a reasonably high acceptance rate:

```scala mdoc
val divergences = nutsChain.numDivergent
val meanAcceptance = nutsChain.infos.map(_.acceptanceRate).sum / nutsChain.size
```

### Running several chains

One chain cannot tell us whether it has explored the whole posterior. The
standard check is to run several chains from different starting points and
compare them. `latent.initRandom` draws a random starting point, as Stan
does. A chain is ordinary code: here a
warmup that tunes NUTS for this chain alone, followed by sampling.

```scala mdoc:silent
def chain(key: Key, start: latent.Params) =
  val (warmupKey, sampleKey) = key.split2()
  val adapted = NUTS.adapt(latent, warmupKey, start)
  adapted.sample(sampleKey, 1000)
```

To run the chains at the same time, we run each in a `Future`:

```scala mdoc:silent
import scala.concurrent.{Await, Future, blocking}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration

val Seq(initKey, chainsKey) = nutsKey.split(2)
val starts = initKey.split(4).map(latent.initRandom(_))
val runs = chainsKey.split(4).zip(starts).map((key, start) => Future(blocking(chain(key, start))))
val chains = Await.result(Future.sequence(runs), Duration.Inf)
```

The summary of all chains pools their draws. Its two diagnostic columns
compare the chains: `rhat` (R-hat) compares the variance between the chains
with the variance within them. Values close to 1 mean they agree, and values
above about 1.01 are a warning. `ess`, the effective sample size, estimates
how many independent draws the correlated draws of the chains are worth:

```scala mdoc
println(summary(chains)(s => (alpha = s.alpha, beta = s.beta, sigma = s.sigma)))
```

The rows of `beta` are numbered by default; `summary(chains, Axis[Feature].names(...))`
names them. The diagnostics on their own, as tensors, are
`Diagnostics.rhat` and `Diagnostics.ess`, both in `witdraw.inference`.

NUTS is not the only inference method: `MAP(latent)` finds the posterior mode,
and `Laplace(latent)` approximates the posterior by a Gaussian around its mode
and is run like NUTS. Both are in `witdraw.inference` and take the same
`latent`.

## Asking the posterior new questions

The posterior draws are draws of $\theta$ only, but the joint distribution
still tells us how data arises given $\theta$. If we fix the parameters to a
posterior draw and run the recipe forwards, we get a new data set that the
model considers plausible after having seen the data. Each pair
$(\theta, y_\text{rep})$ is a draw from the joint distribution of parameters
and new data given the observed data. Ignoring $\theta$, as in the prior
predictive check, leaves draws from the *posterior predictive* distribution
$p(y_\text{rep} \mid y_\text{obs}, x)$.

No new machinery is needed: this is the same model, the same `observe` and the
same `ForwardSampler` as in the prior predictive check. Only the values we fix
are different. Fixing the parameters is plain conditioning here, because they
are drawn before $y$. We use every tenth draw of the chain:

```scala mdoc:silent
val thinned = samples.grouped(10).map(_.head).toSeq
val replicated = replicateKey.split(thinned.size).zip(thinned).map { (key, s) =>
  ForwardSampler.run(key):
    observe(Alpha := s.alpha, Beta := s.beta, Sigma := s.sigma):
      model(xData)
  .y
}
```

A *posterior predictive check* compares the replicated data sets with the
observed one. If the model describes the data well, the observed data should
look like a typical replication. We compare them through a statistic, here the
standard deviation of the outcomes, $T(y) = \mathrm{sd}(y)$, and ask how
often a replication is more spread out than the data:

$$
p = P\big(T(y_\text{rep}) > T(y_\text{obs}) \mid y_\text{obs}, x\big).
$$

As for the posterior mean, the fraction of replications for which this holds
estimates this probability:

```scala mdoc:silent
val replicatedSd = replicated.map(_.std.item)
```

```scala mdoc
val observedSd = yData.std.item
val pValue = replicatedSd.count(_ > observedSd).toFloat / replicatedSd.size
```

A value close to 0 or 1 would mean that the observed spread is unusual under
the model. Since we simulated the data from the model itself, we expect a
value well away from both.

## Composing models

`model` above tells the whole story in one method. But the joint density has
two parts, the prior $p(\theta)$ and the likelihood $p(y \mid \theta, x)$, and
as models grow it helps to write them as separate parts of the program, for
example to try a likelihood with different priors. Since a model is an
ordinary method, this is ordinary Scala: a part of a model is a method with a
`(using ProbContext)` parameter, and calling it from another model runs it in
the same context.

The prior, $p(\theta) = p(\alpha)\, p(\beta)\, p(\sigma)$, samples the
parameters and returns them as a named tuple:

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

The likelihood, $p(y \mid \theta, x)$, takes the parameters as ordinary
tensors. Like the bar in $p(y \mid \theta, x)$, its signature says that it
depends on the parameters, without knowing where they come from:

```scala mdoc:silent
def likelihood(
    x: Tensor2[Sample, Feature, Float32],
    params: Params
)(using ProbContext) =
  val mu = x.dot(Axis[Feature])(params.beta) +! params.alpha
  Y ~ Normal.isotropic(mu, params.sigma)
```

The full model is their product, $p(\theta)\, p(y \mid \theta, x)$: it runs
the prior, then the likelihood with the parameters the prior returned. It
takes the prior as a parameter, so that we can swap it. A
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

## Summary: from mathematics to code

Every step of this guide was a question asked of one joint distribution, and
each has a direct counterpart in the code:

| Mathematics | Code |
| --- | --- |
| The joint distribution $p(\theta, y \mid x)$ | `model(xData)` |
| A draw $(\theta, y) \sim p(\theta, y \mid x)$ | `ForwardSampler.run(key)(model(xData))` |
| A draw $y \sim p(y \mid \theta, x)$ for fixed $\theta$ | `ForwardSampler.run(key)(observe(Alpha := …, …)(model(xData)))` |
| The log-density $\log p(\theta, y \mid x)$ at given values | `ModelScorer.run(observe(Y := …, Alpha := …, …)(model(xData)))` |
| $\theta \mapsto \log p(\theta, y_\text{obs} \mid x)$, the log posterior up to a constant | `latent.logDensity`, with `latent = Latent.discover(observe(Y := yData)(model(xData)))` |
| Draws $\theta^{(s)} \sim p(\theta \mid y_\text{obs}, x)$ | `NUTS.adapt(latent, …).sample(…).samples` |
| Posterior means, standard deviations, credible intervals | `summary(chains)(…)` |
| Draws $y_\text{rep} \sim p(y_\text{rep} \mid y_\text{obs}, x)$ | `ForwardSampler` with `observe` of each posterior draw |
| Prior times likelihood, $p(\theta)\, p(y \mid \theta, x)$ | `regression(xData, prior(nFeatures))`, which runs `prior`, then `likelihood` |
