// Nick Horner
// Naive Bayes in Scala
import data.Data
import naivebayes.NaiveBayes

/** Entry class that exercises the Naive Bayes classifier 
 *  Usage on command line: naivebayes trainingFile testingFile [bin]  
 */
object NaiveTest extends App {
  val fileTrain = if (this.args.size > 0) this.args(0).toString else "optdigits.train"
  val fileTest  = if (this.args.size > 1) this.args(1).toString else "optdigits.test"
  val binning   = if (this.args.size > 2 && this.args(2).toString == "bin") true else false

  lazy val dataTrain = Data.getData(fileTrain)
  lazy val dataTest  = Data.getData(fileTest)

  val classes = Range(0, 10).toArray

  println(s"Size of training data is ${dataTrain.size}")

  val NB = new NaiveBayes(dataTrain, classes, binning) // create NB model with defaults or command line values

  println("Training run:")
  printStats(NB.classify(dataTrain))

  val result = NB.classify(dataTest) // classify test set

  /* Print confusion matrix
   * Example:
   *    0   1   2   3   4   5   6   7   8   9
   * --------------------------------------------
   *   93   0   0   0   2   0   0   0   0   0|  0
   *    0  89   0   0   0   0   1   0   1   0|  1
   *    0   0  91   0   0   0   0   0   3   0|  2
   *    0   0   0 103   0   0   0   0   0   2|  3
   *    0   1   0   0  73   0   0   6   1   3|  4
   *    0   1   0   0   1  79   0   0   2   5|  5
   *    0   2   0   0   0   0  89   0   0   0|  6
   *    0   0   1   0   0   0   0  93   0   0|  7
   *    0   0   0   0   0   0   0   0  87   1|  8
   *    0   0   0   2   2   0   0   5   2  93|  9 
   */
  printConfusionMatrix(result)
  println(s"Binning = $binning")

  println("Testing run:")
  printStats(result)

  /**Helper method that prints out a confusion matrix for a result set
   */
  def printConfusionMatrix(results: Array[(Int, Double, Int)]) {
    println("Confusion Matrix:")
    classes.foreach { x => print(f"${x}%4d") }
    println
    print("-" * 44)
    println
    for (actual <- classes) {
      val actuals = results.filter { case Tuple3(actualClass, _, _) => actualClass == actual }
      for (cl <- classes) {
        val classifiedAs = actuals.filter { case Tuple3(_, _, classifiedClass) => classifiedClass == cl }.size
        print(f"$classifiedAs%4d")
      }
      println(f"|${actual}%3d")
    }
    println("NOTE: actuals form far right column")
  }

  /**Helper method that prints out the stats of a result set
   */
  def printStats(results: Array[(Int, Double, Int)]) {
    val correct = results.filter { case (actualClass, _, classifiedClass) => actualClass == classifiedClass }.size
    val accuracy = correct.toDouble / results.size
    println(s"Resulted in $correct out of ${results.size}, for ${accuracy * 100}% accuracy")
  }
}