package witdraw

import dimwit.*
import dimwit.stats.*

class TransformSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  trait Comp derives Label
  object Mu extends Site[Tensor1[Comp, Float32]]
  object Sigma extends Site[Tensor0[Float32]]

  def model(using ProbContext) =
    val mu = Mu ~ Normal(Tensor1(Axis[Comp] -> 3).fill(0f), Tensor1(Axis[Comp] -> 3).fill(1f))
    val sigma = Sigma ~ HalfNormal(0f, 1f)
    (mu = mu, sigma = sigma)

  def ordered(using ProbContext) =
    transform(Mu.withBijector(Bijector.ordered[Comp]))(model)

  test("the chosen bijector is used: any parameters give increasing values"):
    val latent = Latent.discover(ordered)
    for i <- 0 until 5 do
      val mu = latent.initRandom(Key(i))(Mu).toArray
      assert(mu.sliding(2).forall(p => p(0) < p(1)), mu.mkString(", "))

  test("a bijector to another type does not compile"):
    assert(compileErrors("Sigma.withBijector(Bijector.ordered[Comp])").nonEmpty)

  test("errors: a site the model never samples, a site given twice, nested transforms"):
    object Unused extends Site[Tensor0[Float32]]
    val never = intercept[IllegalArgumentException]:
      Latent.discover(transform(Unused.withBijector(Bijector.exp))(model))
    assert(never.getMessage.contains("never samples it"), never.getMessage)
    val twice = intercept[IllegalArgumentException]:
      Latent.discover(transform(Sigma.withBijector(Bijector.exp), Sigma.withBijector(Bijector.exp))(model))
    assert(twice.getMessage.contains("more than once in the same transform"), twice.getMessage)
    val nested = intercept[IllegalArgumentException]:
      Latent.discover(transform(Sigma.withBijector(Bijector.exp))(transform(Sigma.withBijector(Bijector.exp))(model)))
    assert(nested.getMessage.contains("nested transform"), nested.getMessage)

  test("forward sampling ignores the bijectors"):
    val plain = ForwardSampler.run(Key(1))(model)
    val transformed = ForwardSampler.run(Key(1))(ordered)
    assertEquals(transformed.mu.toArray.toSeq, plain.mu.toArray.toSeq)
