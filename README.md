# WitDraw - Probabilistic programming with wit


## Vision

Probabilistic programs that are easy to read, reason about and trust, even when 
the model becomes complex. The model itself should read like the statistical story it
tells, with every assumption visible in the code. 

## Why

A probabilistic model is a set of assumptions about how the data came to be.
Its conclusions are only as good as those assumptions, so someone has to
understand them and vouch for them. More and more often, that someone did not
write the model but an ai-agent or colleague did. It is thus important that all the
assumptions are in the code and that a compiler helps to check them. 

## How

WitDraw builds on [DimWit](https://github.com/dimwit-dev/dimwit), which makes
tensor dimensions part of Scala 3 types. On top of it:

- **Variables declare their assumptions.** Every random variable is a
  `Site[T]`, whose type states its shape, named dimensions and data type. The
  compiler rejects a model that uses a variable inconsistently with what it
  declares.
- **The model is written once.** A model is a plain Scala function that only
  tells the generative story. Prior sampling, predictive checks and scoring for
  inference are all interpretations of that same code, so there is no second
  version that can drift from the one a reader reviewed.
- **Data is attached, not built in.** `model.observe(Y := data)` conditions a
  model without changing its code, so it is always explicit which variables
  are observed.

The resulting log-densities are differentiable JAX code, ready for NUTS, MAP
estimation or quadratic approximation.

New to WitDraw? The [quickstart](docs/quickstart.md) walks through a complete
Bayesian linear regression, from writing the model to checking the posterior.


## Example

A Bayesian linear regression, the running example of the
[quickstart](docs/quickstart.md). Each random variable is declared as a
site, together with its shape:

```scala
import dimwit.*
import dimwit.stats.*
import witdraw.*

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
def model(x: Tensor2[Sample, Feature, Float32]) = ProbProg:
  val alpha = Alpha ~ Normal(0f, 10f)
  val beta  = Beta ~ Normal.isotropic(Tensor1(x.extent(Axis[Feature])).fill(0f), 10f)
  val sigma = Sigma ~ HalfCauchy(0f, 5f)
  val mu = x.dot(Axis[Feature])(beta) +! alpha   // +! broadcasts the scalar alpha
  val y = Y ~ Normal.isotropic(mu, sigma)
  (alpha = alpha, beta = beta, sigma = sigma, y = y)
```

**Variables declare their assumptions.** If the likelihood doesn't match what
`Y` declares, for example because the features were forgotten, the model
does not compile:

```scala
Y ~ Normal.isotropic(beta, sigma)
// error: Found:    Tensor[Tuple1[Feature], Float32]
//        Required: Tensor[Tuple1[Sample], Float32]
```

**The model is written once.** Given features `xData` and observed outcomes
`yData`, the same definition draws from the prior predictive and gives the
log-density for inference:

```scala
val modelOnX = model(xData)

val priorDraw = ForwardSampler.run(Key(0))(modelOnX)   // a named tuple of all variables
val ySim: Tensor1[Sample, Float32] = priorDraw.y

type PosteriorParams = (alpha: Alpha.Value, beta: Beta.Value, sigma: Sigma.Value)

def logDensity(params: PosteriorParams): Tensor0[Float32] =
  ModelScorer.run(modelOnX.observe(Y := yData).observeAll(params))
```

**Data is attached, not built in.** Observations enter only through
`observe`, and each `:=` is checked against the type of its site.
`observeAll` conditions every site named by a field of `params` (`sigma`
conditions `Sigma`). The model never mentions `yData`, so whether `Y` is
observed is decided where the model is used.

`logDensity` is ready for gradient-based inference. The
[quickstart](docs/quickstart.md) continues from here: it simulates data,
keeps `sigma` positive by sampling it on the log scale, runs NUTS and checks
the posterior against the true parameters.

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
