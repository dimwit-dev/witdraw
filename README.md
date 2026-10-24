# WitDraw - Probabilistic programming with wit

## Vision

Probabilistic programs that are easy to read, reason about and trust, even when
the model becomes complex. The model itself should read like the statistical
story it tells, with every assumption visible in the code.

## Why

A probabilistic model is a set of assumptions about how the data came to be.
Its conclusions are only as good as those assumptions, so someone has to
understand them and vouch for them. More and more often, they did not write
the model themselves; an AI agent or a colleague did. It is thus important that
all the assumptions are in the code and that a compiler helps to check them.

## How

WitDraw builds on [DimWit](https://github.com/dimwit-dev/dimwit), which makes
tensor dimensions part of Scala 3 types: you name what each dimension means,
and the compiler checks that the names are used consistently. WitDraw carries
this over to probabilistic programs:

- Every random variable is a `Site[T]`, whose type states its shape, named
  dimensions and data type. The compiler rejects a model that uses a variable
  inconsistently with what it declares.
- A model is a plain Scala function that only tells the generative story.
  Prior sampling, predictive checks and scoring for inference all run the same
  code, so there is no second version that can drift from the one a reader
  reviewed.
- `observe(Y := data)(model)` conditions a model without changing its code,
  so it is always explicit which variables are observed.
- Models need a `ProbContext` capability to run, in the style of Scala 3's
  direct-style capabilities. Sub-models are ordinary method calls, and
  handlers such as `observe` are ordinary functions that wrap a block.

New to WitDraw? The [quickstart](docs/quickstart.md) walks through a complete
Bayesian linear regression, from writing the model to checking the posterior.

## Example

A Bayesian linear regression. Each random variable is declared as a site,
together with its shape:

```scala
import dimwit.*
import dimwit.stats.*
import witdraw.*
import witdraw.inference.NUTS

trait Sample derives Label   // one entry per observation
trait Feature derives Label  // one entry per feature, i.e. per regression weight

object Alpha extends Site[Tensor0[Float32]]           // intercept
object Beta  extends Site[Tensor1[Feature, Float32]]  // regression weights
object Sigma extends Site[Tensor0[Float32]]           // noise scale (> 0)
object Y     extends Site[Tensor1[Sample, Float32]]   // outcomes
```

The model tells the generative story, and nothing else. The feature matrix
`x` is an input. The outcomes `Y` are a random variable like any other:

```scala
def model(x: Tensor2[Sample, Feature, Float32])(using ProbContext) =
  val alpha = Alpha ~ Normal(0f, 10f)
  val beta  = Beta ~ Normal.isotropic(Tensor1(x.extent(Axis[Feature])).fill(0f), 10f)
  val sigma = Sigma ~ HalfCauchy(0f, 5f)
  val mu = x.dot(Axis[Feature])(beta) +! alpha   // +! broadcasts the scalar alpha
  val y = Y ~ Normal.isotropic(mu, sigma)
  (alpha = alpha, beta = beta, sigma = sigma, y = y)
```

If the likelihood doesn't match what `Y` declares, for example because the
features were forgotten, the model does not compile:

```scala
val y = Y ~ Normal.isotropic(beta, sigma)
// error (simplified):
//        Found:    Tensor[Tuple1[Feature], Float32]
//        Required: Tensor[Tuple1[Sample], Float32]
```

The same model is used for sampling and for scoring. Observations enter through
`observe`, and each `:=` is checked against the type of its site. Whether a
variable is observed is decided where the model is used, not where it is
defined.

```scala
// draw every variable from the prior: a named tuple (alpha, beta, sigma, y)
val prior = ForwardSampler.run(Key(0))(model(xData))

// draw with the noise scale fixed to 1
val withKnownSigma = ForwardSampler.run(Key(1)):
  observe(Sigma := Tensor0(1f)):
    model(xData)
```

For inference, `Latent.discover` traces one run of the observed model and
finds its latent sites, here `alpha`, `beta` and `sigma`. It gives a parameter
type for them and a starting point, and the inference methods in
`witdraw.inference` work on it directly: `NUTS` samples the posterior, `MAP`
finds its mode, and `Laplace` approximates it by a Gaussian. Constraints are
declared per site:

```scala
val posterior = Latent.discover(Constraint(Sigma, Bijector.exp)):
  observe(Y := yData)(model(xData))

// warmup tunes the step size and the mass matrix
val adapted = NUTS.adapt(posterior, posterior.init, Key(3))
val draws = adapted.sampler.run(adapted.position, Key(4)).take(2000).map(_._1).toList
draws.map(_(Sigma))   // posterior draws of sigma, each a Sigma.Value (> 0)
```

The [quickstart](docs/quickstart.md) is a complete, runnable version of this
example: it simulates data, runs NUTS and checks the posterior against the
true parameters. It also shows how larger models are built from parts,
including components: parts that own their sites, so that a part can be used
more than once in the same model.

More examples, including ports of models from
[Statistical Rethinking](https://xcelab.net/rm/) and the Stan user guide, are
in [examples/](examples/src/main/scala/witdraw/examples).

## Using WitDraw as a Library

### Requirements

- Scala 3.9+
- sbt 1.11+
- Python 3.12 (managed via [uv](https://docs.astral.sh/uv/))

### Installation

WitDraw is currently published as a snapshot. Add the snapshot repository and
the dependency to your `build.sbt`:

```scala
resolvers += "Central Portal Snapshots" at "https://central.sonatype.com/repository/maven-snapshots/"
libraryDependencies += "ch.contrafactus" %% "witdraw-core" % "0.1-SNAPSHOT"
```

### Python setup

WitDraw runs on JAX through DimWit. Create a `pyproject.toml` with the
required Python packages:

```toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = "==3.12.*"
dependencies = [
    "einops>=0.8.1",
    "jax>=0.8.1",       # or "jax[cuda12]" for GPU support
    "blackjax>=1.5.0"
]
```

Then call `dimwit.initialize()` at the start of your program. See the
[DimWit README](https://github.com/dimwit-dev/dimwit) for details on configuring
the Python environment manually.

### Running the examples

```bash
sbt examples/run   # pick an example from the list
```

## Status

WitDraw is in an early, experimental stage. The core ideas (typed sites,
context-interpreted programs, conditioning) are taking shape, but the API is
evolving quickly and the library is not ready for production use.

## Contributing

[Issues](https://github.com/dimwit-dev/witdraw/issues) and
[pull requests](https://github.com/dimwit-dev/witdraw/pulls) are welcome.

## License

WitDraw is licensed under the [Apache License 2.0](LICENSE).
