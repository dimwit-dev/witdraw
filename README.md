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


## Example

A linear regression. Each random variable is declared as a site, together
with its shape:

```scala
import dimwit.*
import dimwit.stats.*
import witdraw.*

trait Data derives Label       // one entry per observation
trait Predictor derives Label  // one entry per regression coefficient

object Alpha extends Site[Tensor0[Float32]]             // intercept
object Beta  extends Site[Tensor1[Predictor, Float32]]  // coefficients
object Sigma extends Site[Tensor0[Float32]]             // noise scale
object Y     extends Site[Tensor1[Data, Float32]]       // outcome
```

The model tells the generative story, and nothing else. The covariates `x`
are an input. The outcome `Y` is a random variable like any other:

```scala
def model(x: Tensor2[Data, Predictor, Float32]) = ProbProg:
  val alpha = Alpha ~ Normal(0f, 100f)
  val beta  = Beta ~ Normal.isotropic(Tensor1(x.extent(Axis[Predictor])).fill(0f), 100f)
  val sigma = Sigma ~ HalfCauchy(0f, 25f)
  val mu = x.dot(Axis[Predictor])(beta) +! alpha   // +! broadcasts the scalar alpha
  Y ~ Normal.isotropic(mu, sigma)
```

**Variables declare their assumptions.** If the likelihood doesn't match what
`Y` declares, for example because the covariates were forgotten, the model
does not compile:

```scala
Y ~ Normal.isotropic(beta, sigma)
// error: Found:    Tensor[Tuple1[Predictor], Float32]
//        Required: Tensor[Tuple1[Data], Float32]
```

**The model is written once.** Given covariates `x` and observed outcomes
`y`, the same definition draws from the prior predictive and gives the
log-density for inference:

```scala
val ySim: Tensor1[Data, Float32] = ForwardSampler.run(Key(0))(model(x))

type Posterior = (alpha: Tensor0[Float32], beta: Tensor1[Predictor, Float32], sigma: Tensor0[Float32])

def logDensity(p: Posterior): Tensor0[Float32] =
  ModelScorer.run(
    model(x).observe(Y := y, Alpha := p.alpha, Beta := p.beta, Sigma := p.sigma)
  )
```

**Data is attached, not built in.** Observations enter only through
`observe`, and each `:=` is checked against the type of its site. The model
above never mentions `y`, so whether `Y` is observed is decided where the
model is used. `logDensity` can be handed directly to a sampler, for example
`NUTSAlgorithm[Posterior](stepSize = 0.1)(logDensity)`.

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
