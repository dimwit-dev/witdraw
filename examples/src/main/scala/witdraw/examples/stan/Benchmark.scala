package witdraw.examples.stan

import dimwit.*
import witdraw.*
import witdraw.inference.{NUTS, NUTSInfo, NUTSState, SamplingAlgorithm}

/** Runs WitDraw's NUTS on one of the Stan examples, for the comparison with
  * CmdStan driven by `examples/benchmark/bench.py`.
  *
  * {{{
  * sbt "examples/runMain witdraw.examples.stan.stanBenchmark <model> <data.json> <stepSize> <outDir> <nWarmup> <nSamples>"
  * }}}
  *
  * `<model>` is `linreg`, `mixture` or `hmm`; the data file is the one given to
  * Stan. `<stepSize>` is either a fixed step size, used with the identity mass
  * matrix and no adaptation, or `adapt`, to tune both with [[NUTS.adapt]].
  * Writes `<outDir>/witdraw_<model>_draws.csv` (constrained draws after
  * warmup, columns named as by CmdStan) and `<outDir>/witdraw_<model>_info.json`
  * (timings and per-draw integration steps).
  *
  * Compiling is timed apart from running: each compiled part runs twice, and
  * the second run is the one timed. For the warmup, the second run reuses the
  * first one's executable only through JAX's persistent compilation cache,
  * which `bench.py` turns on.
  */
object Benchmark:

  private def floats(js: ujson.Value): Array[Float] =
    js.arr.map(_.num.toFloat).toArray

  /** CmdStan-style columns `name[1]`, ..., `name[n]` of a vector-valued site. */
  private def vector[P](name: String, n: Int)(values: P => Array[Float]) =
    (0 until n).map(i => s"$name[${i + 1}]" -> ((p: P) => values(p)(i)))

  private def secs(a: Long, b: Long) = (b - a) / 1e9

  /** Runs `f` twice and returns the second result, the time of the first run
    * minus that of the second (the compile time), and the time of the second.
    */
  private def twice[A](f: Int => A): (A, Double, Double) =
    val t0 = System.nanoTime()
    f(0)
    val t1 = System.nanoTime()
    val result = f(1)
    val t2 = System.nanoTime()
    (result, secs(t0, t1) - secs(t1, t2), secs(t1, t2))

  @main
  def stanBenchmark(
      model: String,
      dataPath: String,
      stepSize: String,
      outDir: String,
      nWarmup: Int,
      nSamples: Int
  ): Unit =
    dimwit.initialize()
    val data = ujson.read(os.read(os.Path(dataPath, os.pwd)))
    val out = os.Path(outDir, os.pwd)
    os.makeDir.all(out)
    val adapt = stepSize == "adapt"

    /** Samples `latent` from `start` and writes the draws and timings. */
    def bench(latent: Latent, setupStart: Long)(
        start: latent.Params,
        columns: Seq[(String, latent.Params => Float)]
    ): Unit =
      val tSetup = System.nanoTime()
      type Sampler = SamplingAlgorithm[latent.Params, NUTSState[latent.Params], NUTSInfo]

      // the sampler, where it starts, how many draws it makes and how many of
      // them are warmup, the step size, and the warmup's compile and run time
      val (sampler, from, numDraws, numWarmupDraws, step, warmupCompile, warmupRun) =
        if adapt then
          val (a, compile, run) =
            twice(i => NUTS.adapt(latent, start, Key(6L + i), numSteps = nWarmup))
          (a.sampler: Sampler, a.position, nSamples, 0, a.stepSize, compile, run)
        else
          val s = stepSize.toDouble
          // no adaptation: the warmup is the first nWarmup draws of the loop
          (NUTS(latent, s): Sampler, start, nWarmup + nSamples, nWarmup, s, 0.0, 0.0)

      val (all, samplingCompile, samplingRun) =
        twice(i => sampler.sample(from, Key(8L + i), numDraws))

      val steps = all.map(_._2.numIntegrationSteps)
      val kept = all.drop(numWarmupDraws)
      val divergences = kept.count(_._2.isDivergent)

      // quoted, since names like gamma[1,2] contain commas
      val header = columns.map((name, _) => s"\"$name\"").mkString(",")
      val rows = kept.map((p, _) => columns.map((_, f) => f(p).toString).mkString(","))
      os.write.over(
        out / s"witdraw_${model}_draws.csv",
        (header +: rows).mkString("", "\n", "\n")
      )

      val compile = warmupCompile + samplingCompile
      val info = ujson.Obj(
        "model" -> model,
        "adapted" -> adapt,
        "stepSize" -> step,
        "nWarmup" -> nWarmup,
        "nSamples" -> nSamples,
        "setupSeconds" -> secs(setupStart, tSetup),
        "compileSeconds" -> compile,
        "warmupSeconds" -> warmupRun,
        // the timed sampling loop: with a fixed step size, warmup and sampling
        "samplingSeconds" -> samplingRun,
        // the integration steps of each draw of the timed sampling loop
        "leapfrogSteps" -> ujson.Arr.from(steps),
        "divergences" -> divergences
      )
      os.write.over(out / s"witdraw_${model}_info.json", ujson.write(info, 2))
      println(
        f"$model: setup ${secs(setupStart, tSetup)}%.2fs, compile $compile%.2fs, " +
          f"warmup $warmupRun%.2fs, sampling loop $samplingRun%.2fs ($numDraws draws), " +
          f"step size $step%.4f, ${steps.sum} leapfrog steps in the loop, $divergences divergences"
      )

    val t0 = System.nanoTime()
    model match
      case "linreg" =>
        import LinearRegression.*
        val x = Tensor2(Axis[Data], Axis[Predictor])
          .fromArray(data("x").arr.map(floats).toArray)
        val y = Tensor1(Axis[Data]).fromArray(floats(data("y")))
        val l = latent(x, y)
        bench(l, t0)(
          l.init,
          Seq("alpha" -> ((p: l.Params) => p(Alpha).item)) ++
            vector[l.Params]("beta", data("K").num.toInt)(_(Beta).toArray) :+
            ("sigma" -> ((p: l.Params) => p(Sigma).item))
        )

      case "mixture" =>
        import Mixture.*
        val k = data("K").num.toInt
        val y = Tensor1(Axis[Data]).fromArray(floats(data("y")))
        val l = latent(y, k)
        // mu must start inside its ordered constraint
        val muStart = Array.tabulate(k)(i => i.toFloat - (k - 1) / 2f)
        bench(l, t0)(
          l.initWith(Mu := Tensor1(Axis[Comp] -> k).fromArray(muStart)),
          vector[l.Params]("theta", k)(_(Theta).toArray) ++
            vector[l.Params]("mu", k)(_(Mu).toArray) ++
            vector[l.Params]("sigma", k)(_(Sigma).toArray)
        )

      case "hmm" =>
        import HMM.*
        val y = Tensor1(Axis[Time]).fromArray(floats(data("y")))
        val l = latent(y)
        val gamma =
          for i <- 0 until 3; j <- 0 until 3
          yield s"gamma[${i + 1},${j + 1}]" ->
            ((p: l.Params) => p(Gamma).toArray(i)(j))
        bench(l, t0)(
          l.init,
          vector[l.Params]("mu", 3)(_(Mu).toArray) ++
            Seq("sigma" -> ((p: l.Params) => p(Sigma).item)) ++
            vector[l.Params]("rho", 3)(_(Rho).toArray) ++
            gamma
        )

      case other =>
        throw IllegalArgumentException(s"unknown model '$other'")
