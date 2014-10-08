package naivebayes
//import Math.log
import scala.collection.mutable

/** Workhorse of our Bayesian classifier
 * 
 * @constructor creates a Bayesian classifier with a given training set
 *              and initializes the model
 * @param trainingSet the data to train our classifier on
 * @param classes the possible classfications
 * @param binning should the classes be binned
 */

class NaiveBayes(trainingSet: Array[Array[Int]], classes: Array[Int], binning: Boolean = false) {
  type Probability = Double
  val trainingSize    = trainingSet.size
  val numAttributes   = trainingSet.head.size - 1 // the last element is the actual classification
  val attributeValues = if(!binning) Range(0, 17).toArray
                        else Array(Range(0, 5).toArray, Range(5, 9).toArray, Range(9, 13).toArray, Range(13, 17).toArray) //bins

  assert(trainingSize > 0, "Training set empty")
  val numValues = attributeValues.size
  assert((binning && numValues == 4) || numValues == 17, "Error in sizing attributes")
  
  var model = Map[String, Probability]() // hold our model
  
  // A. compute prior probability for each class ie. P('1')
  computePForClasses()
  // B. compute conditional probabilities for each digits and for each value of each attribute. 
  // ie. P(Attribute1=0|'1') is the probability of that Attribute 1 has value 0, given that the class is digit '1' 
  // C. smooth the conditional probabilities using Laplace (add-one) smoothing
  for(cls <- classes){
    	computePForEachAttribute(cls) // calculate for all classes
  }
  /** computes prior probability for all classes
   * 
   * i.e. P('1')
   */
  private def computePForClasses() {
    for(cls <- classes){
      val count = trainingSet.filter(x => x.last == cls).size
      val P: Probability = count.toDouble / trainingSize
      model += Tuple2(cls.toString, P)
    }
    assert(model.size == classes.size, "Error creating model for classes")
  }
  
  /** computes prior probability for each attribute, adding Laplace smoothing
   * 
   * i.e. P(Attribute1=0|'1') is inserted into the model with the key: 1=0|1
   * @param cls class to compute
   */
  private def computePForEachAttribute(cls: Int) {
    val clsData = trainingSet.filter(x => x.last == cls)
    
    for(attIdx <- 0 to numAttributes){
      for(value <- attributeValues){   
        val (key, attValData) = value match {
          case idx: Int => (s"$attIdx=$idx|$cls", clsData.filter(x => x(attIdx) == value))
          case arr: Array[Int] => (s"$attIdx=${arr.mkString}|$cls", clsData.filter(x => arr.contains(x(attIdx))))
          case _ => throw new Exception("Attributes were improperly formed")
        }
                 
        model += Tuple2(key, (attValData.size + 1.0) / (clsData.size + numValues))
      }    
    }
  }
  /* type aliases for classification */
  type ActualClass = Int
  type ConfidenceLevel = Double
  type ClassifiedClass = Int
  
  /** classify based on the training data given
   * 
   * calculate confidence interval for all classes, then find class that maximizes
   * @param testSet Data set to classify given the model
   * @return array of classifications where each element is a Tuple of (ActualClass, Confidence, Classification)
   */
  def classify(testSet: Array[Array[Int]]): Array[(ActualClass, ConfidenceLevel, ClassifiedClass)] = {
    var result = Array[(ActualClass, ConfidenceLevel, ClassifiedClass)]()
    for(test <- testSet){
	    val actualClass: ActualClass = test.last
	    var maxClass: (ClassifiedClass, ConfidenceLevel) = (0, 0.0)
	    var maxInit = false
	    
	    for(cls <- classes){
	      var confidence: ConfidenceLevel = Math.log(model.getOrElse(cls.toString, throw new Exception("shouldn't reach"))) // log P(class)
	      for(attIdx <- 0 to numAttributes){
	        //form value
	        val vall: String = if(!binning) test(attIdx).toString //TODO refactor into a match
	                   else {       
	                     findAttribute(test, attIdx)
	                   }
	        val key = s"$attIdx=${vall}|$cls"
	        confidence += Math.log(model.getOrElse(key, throw new Exception("shouldn't reach"))) // + log(P(x_n|class)
	      }
	      // confidence calculated for given class
	      if(confidence > maxClass._2 || !maxInit){ // found new prospective class
	        maxClass = Tuple2(cls, confidence)
	        maxInit = true
	      }
	    }
	    val el: (ActualClass, ConfidenceLevel, ClassifiedClass) = Tuple3(actualClass, maxClass._2, maxClass._1)
	    result = result :+ el
    }
    result
  }
  
  /** A helper method for classify
   * 
   * Finds attribute when binning is done
   * @return 
   */
  private def findAttribute(test: Array[Int], attIdx: Int): String = {
    for (i <- 0 until attributeValues.asInstanceOf[Array[Array[Int]]].size) {
      if (attributeValues.asInstanceOf[Array[Array[Int]]](i).contains(test(attIdx))) {
        return attributeValues.asInstanceOf[Array[Array[Int]]](i).mkString
      }
    }
    throw new Exception("shouldn't reach")
  }
}