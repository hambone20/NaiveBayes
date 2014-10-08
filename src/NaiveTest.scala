// Nick Horner
// Naive Bayes in Scala
import data.Data
import naivebayes.NaiveBayes

/** Entry class that exercises the Naive Bayes classifier */
object NaiveTest extends App {
  val trainFile = if (this.args.size > 0) this.args(0).toString else "optdigits.train"
  val testFile  = if (this.args.size > 1) this.args(1).toString else "optdigits.test"
  val binning   = if (this.args.size > 2 && this.args(2).toString == "bin") true else false

  lazy val trainSet = Data.getData(trainFile)
  lazy val testSet  = Data.getData(testFile)

  val classes = Range(0, 10).toArray //(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

  println(s"Size of trainData is ${trainSet.size}")

  val NB = new NaiveBayes(trainSet, classes, binning) // create NB model with defaults or command line values

  val resultTrain = NB.classify(trainSet)
  val correctTrain = resultTrain.filter{case (actualClass: Int, _, classifiedClass: Int) => actualClass == classifiedClass}.size
  println(s"Training Set: $correctTrain out of ${resultTrain.size}, for ${correctTrain.toDouble / resultTrain.size} accuracy")

  val result = NB.classify(testSet)

  /* Create confusion matrix
   * Ex.
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
  println(s"Binning = $binning")
  println("Confusion Matrix:")
  classes.foreach { x => print(f"${x}%4d") }
  println
  print("-" * 44)
  println
  for (actual <- classes) {
    val actuals = result.filter{case (actualClass, _, _) => actualClass == actual}
    for (cl <- classes) {
      val classifiedAs = actuals.filter{case (_, _, classifiedClass) => classifiedClass == cl}.size
      print(f"$classifiedAs%4d")
    }
    println(f"|${actual}%3d")
  }
  println("NOTE: actuals form far right column")
  val correct = result.filter{case (actualClass, _, classifiedClass) => actualClass == classifiedClass}.size
  println(s"Testing Set: $correct out of ${result.size}, for ${correct.toDouble / result.size} accuracy")
}