# WitDraw - Probabilistic programming with wit

## Vision

Probabilistic programs that are easy to read, express the statistical story they tell, and remain manageable and correct even when the model becomes complex.

## Why

Understanding a statistical model is hard. Expressing the statistical story in code, 
having the compiler help to check that the quantities are used consistently, and being able to run the same code for prior sampling, predictive checks and inference, all make it easier to understand and maintain a model.


## How

A WitDraw model is a joint distribution over all quantities of a problem,
written as the story of how the data is generated. Everything else is a
question asked of that one distribution, by running the same code in
different ways.

As an example, take a Bayesian linear regression. Each random variable is a
*site*, whose type states its shape, named dimensions (through
[DimWit](https://github.com/dimwit-dev/dimwit)) and data type:

```scala
trait Sample derives Label   // one entry per observation
trait Feature derives Label  // one entry per feature

object Alpha extends Site[Tensor0[Float32]]           // intercept
object Beta  extends Site[Tensor1[Feature, Float32]]  // weights
object Sigma extends Site[Tensor0[Float32]]           // noise scale
object Y     extends Site[Tensor1[Sample, Float32]]   // outcomes
```

The model reads like its mathematical definition. The features `x` are an
input; the outcomes are a random variable like the parameters:

```scala
def model(x: Tensor2[Sample, Feature, Float32])(using ProbContext) =
  val alpha = Alpha ~ Normal(0f, 10f)
  val beta  = Beta ~ Normal.isotropic(Tensor1(x.extent(Axis[Feature])).fill(0f), 10f)
  val sigma = Sigma ~ HalfCauchy(0f, 5f)
  val mu = x.dot(Axis[Feature])(beta) +! alpha   // +! broadcasts the scalar alpha
  val y = Y ~ Normal.isotropic(mu, sigma)
  (alpha = alpha, beta = beta, sigma = sigma, y = y)
```

A likelihood that does not match what `Y` declares, e.g. because the features
were forgotten, does not compile:

```scala
val y = Y ~ Normal.isotropic(beta, sigma)
// error (simplified):
//        Found:    Tensor[Tuple1[Feature], Float32]
//        Required: Tensor[Tuple1[Sample], Float32]
```

The same model answers every question. Which variables are observed is decided
where the model is used, not where it is defined:

```scala
// a draw of all variables from the joint distribution: (alpha, beta, sigma, y)
val draw = ForwardSampler.run(Key(0))(model(xData))

// the posterior given the observed outcomes: Latent finds the unobserved
// sites (alpha, beta, sigma), and NUTS draws from their posterior
val latent = Latent.discover:
  observe(Y := yData)(model(xData))
val chain = NUTS.adapt(latent, Key(1), latent.init).sample(Key(2), 2000)

chain.samples.map(_.sigma)   // posterior draws of sigma
println(summary(Seq(chain))(s => (alpha = s.alpha, beta = s.beta, sigma = s.sigma)))
```

The [quickstart](docs/quickstart.md) develops this example step by step, and
explains how each step corresponds to the mathematics: drawing from the joint
distribution, conditioning, inference, posterior predictive checks, and
building models from parts.

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
