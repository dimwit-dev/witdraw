package witdraw

import dimwit.*
import dimwit.stats.*
import dimwit.tensortree.TreeOf.ops.*

class ComponentSuite extends munit.FunSuite:

  override def beforeAll(): Unit = dimwit.initialize()

  trait Time derives Label

  /** One media channel: a positive effect per unit of spend. */
  trait Channel extends Component:
    object Effect extends Site[Tensor0[Float32]]

    def apply(spend: Tensor1[Time, Float32])(using ProbContext) =
      spend *! (Effect ~ HalfNormal(0f, 1f))

    def values(v: SiteValues) = Channel.Values(v(Effect))
    def observations(x: Channel.Values) = Seq(Effect := x.effect)

  object Channel:
    case class Values(effect: Tensor0[Float32])

  trait Media extends Component:
    object tv extends Channel
    object radio extends Channel
    object Noise extends Site[Tensor0[Float32]]

    def values(v: SiteValues) =
      Media.Values(tv.values(v), radio.values(v), v(Noise))
    def observations(x: Media.Values) =
      tv.observations(x.tv) ++ radio.observations(x.radio) :+ (Noise := x.noise)

  object Media:
    case class Values(
        tv: Channel.Values,
        radio: Channel.Values,
        noise: Tensor0[Float32]
    )

  object Y extends Site[Tensor1[Time, Float32]]

  object media extends Media
  object solo extends Channel
  import media.{tv, radio, Noise}

  lazy val tvSpend = Tensor1(Axis[Time]).fromArray(Array(1f, 2f, 3f))
  lazy val radioSpend = Tensor1(Axis[Time]).fromArray(Array(0.5f, 0f, 1f))
  lazy val yData = Tensor1(Axis[Time]).fromArray(Array(1.2f, 2.1f, 3.5f))

  def model(using ProbContext) =
    val contribution = tv(tvSpend) + radio(radioSpend)
    val noise = Noise ~ HalfNormal(0f, 1f)
    Y ~ Normal.isotropic(contribution, noise)

  test("sites are named by the path of their components"):
    assertEquals(tv.Effect.name, "media.tv.Effect")
    assertEquals(radio.Effect.name, "media.radio.Effect")
    assertEquals(Noise.name, "media.Noise")
    assertEquals(solo.Effect.name, "solo.Effect")
    assertEquals(media.path, List("media"))
    assertEquals(tv.path, List("media", "tv"))
    assertEquals(Y.name, "Y")
    assertEquals(NormalModel.Mu.name, "Mu")

  test("an instance that is not an object is named explicitly"):
    val regions = (0 until 2).map: i =>
      new Channel:
        override def name = s"region$i"
    assertEquals(regions.map(_.Effect.name), Seq("region0.Effect", "region1.Effect"))

  test("a component can be used several times, each with its own sites"):
    assertNotEquals[SiteDef[?, ?], SiteDef[?, ?]](tv.Effect, radio.Effect)
    val (_, tr) = ForwardSampler.run(Key(0))(withTrace(model))
    assertEquals(tr.sites, Seq(tv.Effect, radio.Effect, Noise, Y))
    assertEquals(
      tr.toString,
      "Trace(media.tv.Effect, media.radio.Effect, media.Noise, Y)"
    )

  test("sites of components work with observe and Latent"):
    val latent = Latent.discover(
      Constraint(tv.Effect, Bijector.exp),
      Constraint(radio.Effect, Bijector.exp),
      Constraint(Noise, Bijector.exp)
    )(observe(Y := yData)(model))
    assertEquals(latent.sites, Seq(tv.Effect, radio.Effect, Noise))
    val p = latent.initWith(
      tv.Effect := Tensor0(2f),
      radio.Effect := Tensor0(0.5f),
      Noise := Tensor0(1.5f)
    )
    val expected = ModelScorer.run:
      observe(
        tv.Effect := Tensor0(2f),
        radio.Effect := Tensor0(0.5f),
        Noise := Tensor0(1.5f),
        Y := yData
      )(model)
    assertEqualsFloat(latent.modelLogDensity(p).item, expected.item, 1e-4f)

  test("error messages name sites by their path"):
    val e = intercept[IllegalArgumentException]:
      ForwardSampler.run(Key(0)):
        withTrace:
          tv(tvSpend)
          tv(tvSpend)
    assert(e.getMessage.contains("'media.tv.Effect'"), e.getMessage)

  def mediaValues(tv: Float, radio: Float, noise: Float) =
    Media.Values(Channel.Values(Tensor0(tv)), Channel.Values(Tensor0(radio)), Tensor0(noise))

  test("a component reads its values as a typed record, from a trace"):
    val (_, tr) = ForwardSampler.run(Key(0))(withTrace(model))
    val v = media.values(tr)
    assertEquals(v.tv.effect.item, tr(tv.Effect).item)
    assertEquals(v.noise.item, tr(Noise).item)

  test("a component reads its values as a typed record, from Latent"):
    val latent = Latent.discover(
      Constraint(tv.Effect, Bijector.exp),
      Constraint(radio.Effect, Bijector.exp),
      Constraint(Noise, Bijector.exp)
    )(observe(Y := yData)(model))
    // the record also gives the observations, e.g. for a starting point
    val p = latent.initWith(media.observations(mediaValues(2f, 0.5f, 1.5f))*)
    val v = media.values(latent.values(p))
    assertEqualsFloat(v.tv.effect.item, 2f, 1e-5f)
    assertEqualsFloat(v.radio.effect.item, 0.5f, 1e-5f)
    assertEqualsFloat(v.noise.item, 1.5f, 1e-5f)

  test("a record of values is a tensor tree, e.g. to average draws"):
    val draws = List(mediaValues(1f, 2f, 3f), mediaValues(3f, 4f, 5f))
    val mean = Tensor0(1f / draws.size) **! draws.reduce(_ ++ _)
    assertEqualsFloat(mean.tv.effect.item, 2f, 1e-6f)
    assertEqualsFloat(mean.radio.effect.item, 3f, 1e-6f)
    assertEqualsFloat(mean.noise.item, 4f, 1e-6f)
