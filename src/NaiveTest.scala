// Nick Horner
// HW4 - Naive Bayes
import data.Data
import naivebayes.NaiveBayes

object NaiveTest extends App{
	  var trainFile = "optdigits.train"
	  if(this.args.size > 0){ // train
	    trainFile = this.args(0).toString
	  }
	  
	  var testFile = "optdigits.test"
	  if(this.args.size > 1){ // test
	    testFile = this.args(1).toString
	  }
	  
	  var binning = false
	  if(this.args.size > 2 && this.args(2).toString == "bin"){
	    binning = true
	  }
	lazy val trainSet = Data.getData(trainFile)
	lazy val testSet  = Data.getData(testFile)
	
	val classes = Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

	println(s"Size of trainData is ${trainSet.size}")
	
	val NB = new NaiveBayes(trainSet, classes, binning) // create NB model
	
	val resultTrain = NB.classify(trainSet)
    val correctTrain = resultTrain.filter(x => x._1 == x._3).size
	println(s"Training Set: $correctTrain out of ${resultTrain.size}, for ${correctTrain.toDouble / resultTrain.size} accuracy")
	
	val result = NB.classify(testSet)
	
	// create confusion matrix
	println(s"Binning = $binning")
	println("Confusion Matrix:")
	classes.foreach{x => print(f"${x}%4d")}
	println
	for(_ <- 0 to 44)print("-")
	println
	for(i <- classes){
	  val actuals = result.filter(x => x._1 == i)
	  for(j <- classes){
	      print(f"${actuals.filter(x => x._3 == j).size}%4d")
	  }
	  println(f"|${i}%3d")
	}
	println("NOTE: actuals form far right column")
	val correct = result.filter(x => x._1 == x._3).size
	println(s"Testing Set: $correct out of ${result.size}, for ${correct.toDouble / result.size} accuracy")
}