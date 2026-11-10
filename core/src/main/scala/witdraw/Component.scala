package witdraw

/** Where a site or a component is declared: the path of names of the
  * [[Component]]s around it. Sites declared outside any component are in
  * [[Scope.root]], with an empty path.
  */
trait Scope:
  def path: List[String]

object Scope:
  /** The scope outside all components. */
  given root: Scope with
    def path = Nil

/** A part of a model that owns its sites, written as a module: a trait
  * declares the part, and each object that extends it is one instance, with
  * its own sites.
  * {{{
  * trait Linear extends Component:
  *   object Intercept extends Site[Tensor0[Float32]]
  *   def apply(x: Tensor2[Patient, Feature, Float32])(using ProbContext) = ...
  *
  * trait Outcome extends Component:
  *   object mean extends Linear
  *   object scale extends Linear
  *
  * object bloodPressure extends Outcome
  * bloodPressure.mean.Intercept.name   // "bloodPressure.mean.Intercept"
  * }}}
  *
  * A part can return a typed record of its values with its output, so that
  * the model's result, and so every prior and posterior sample, carries it:
  * {{{
  * trait Linear extends Component:
  *   object Intercept extends Site[Tensor0[Float32]]
  *   object Weights extends Site[Tensor1[Feature, Float32]]
  *   def apply(x: Tensor2[Patient, Feature, Float32])(using ProbContext) =
  *     val intercept = Intercept ~ Normal(0f, 1f)
  *     val weights = Weights ~ ...
  *     (output = x.dot(Axis[Feature])(weights) +! intercept, values = Linear.Values(intercept, weights))
  *
  * object Linear:
  *   case class Values(intercept: Tensor0[Float32], weights: Tensor1[Feature, Float32])
  * }}}
  * A record of tensors like `Linear.Values` is a `TensorTree`, so it can be
  * averaged, saved or mapped over as a whole.
  *
  * A component is named like a site, after its object, and is inside the
  * component in whose body it is declared. Parameters of a part are trait
  * parameters, e.g. `trait Linear(priorScale: Float)` and
  * `object mean extends Linear(1f)`.
  *
  * An instance that is not an object, e.g. one of several created in a loop,
  * needs an explicit name: `new Linear { override def name = s"region$i" }`.
  */
trait Component(using parent: Scope) extends Scope:
  def name: String = getClass.getSimpleName.stripSuffix("$")
  def path: List[String] = parent.path :+ name

  /** Sites and components declared in this component's body are inside it. */
  protected given Scope = this

  override def toString: String = s"Component(${path.mkString(".")})"
