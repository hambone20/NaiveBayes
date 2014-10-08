package naivebayes
import Math.log
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
  // Mapping will contain P on class 
  
  val trainingSize    = trainingSet.size
  val numAttributes   = trainingSet.head.size - 1 // the last element is the actual classification
  val attributeValues = if(!binning) Range(0, 17).toArray
                        else Array(Range(0, 5).toArray, Range(5, 9).toArray, Range(9, 13).toArray, Range(13, 17).toArray) //bins

  assert(trainingSize > 0, "Training set empty")
  val numValues = attributeValues.size
  assert((binning && numValues == 4) || numValues == 17, "Error in sizing attributes")
  
  var model = Map[String, Double]() // hold our model
  
  // A. compute prior probability for each class ie. P('1')
  computePForClasses()
  // B. compute conditional probabilities for each digits and for each value of each attribute. 
  // ie. P(Attribute1=0|'1') is the probability of that Attribute 1 has value 0, given that the class is digit '1' 
  // C. smooth the conditional probabilities using Laplace (add-one) smoothing
  for(cls <- classes){
    	computePForEachAttribute(cls) // calc for given class
  }
  /** computes prior probability for all classes
   * 
   * i.e. P('1')
   */
  def computePForClasses() {
    for(cls <- classes){
      val count = trainingSet.filter(_.last == cls).size
      val P = count.toDouble / trainingSize
      model += Tuple2(cls.toString, P)
    }
  }
  
  /** computes prior probability for each attribute
   * 
   * i.e. P(Attribute1=0|'1')
   * @param cls class to compute
   */
  def computePForEachAttribute(cls: Int) { // adjust for Laplace smoothing
    val clsData = trainingSet.filter(x => x.last == cls)
    // attributes 0-16, numValues = 17 of them
    for(attIdx <- 0 to numAttributes){
      for(value <- attributeValues){ // we have the form P(att{attIdx}={value}|cls)
        val key: String = if(!binning) s"$attIdx=$value|$cls"
        		          else s"$attIdx=${(value.asInstanceOf[Array[Int]]).mkString}|$cls" 
        val attValData = if(!binning) clsData.filter(x => x(attIdx) == value) // data that has the given attribute value
                         else{
                        	 clsData.filter(x => value.asInstanceOf[Array[Int]].contains(x(attIdx)))
                         }
                 
        model += Tuple2(key, (attValData.size + 1.0) / (clsData.size + numValues))
      }    
    }
  }
  
  /** classify based on the training data given
   * 
   * @param testSet Data set to classify given the model
   * @return array of classifications 
   */
  def classify(testSet: Array[Array[Int]]) = {
    // calculate confidence interval for all classes, find class that maximizes
    var result = mutable.ArrayBuffer[(Int, Double, Int)]()
    for(test <- testSet){
	    var maxClass = (0, 0.0) // class, confidence
	    var initMax = false
	    val actualClass = test.last
	    for(cls <- classes){
	      var confidence = log(model.getOrElse(cls.toString, throw new Exception("shouldn't reach"))) // log P(class)
	      for(attIdx <- 0 to numAttributes){
	        //form value
	        val vall: String = if(!binning) test(attIdx).toString
	                   else {
	                     def findAttribute() :String = {
	                    	 for(i <- 0 until attributeValues.asInstanceOf[Array[Array[Int]]].size){
	                           if(attributeValues.asInstanceOf[Array[Array[Int]]](i).contains(test(attIdx))){
	                             return attributeValues.asInstanceOf[Array[Array[Int]]](i).mkString 
	                           }
	                         }
	                    	 throw new Exception("shouldn't reach")
	                     }
	                     findAttribute()
	                   }
	        confidence += log(model.getOrElse(s"$attIdx=${vall}|$cls", throw new Exception("shouldn't reach"))) // + log(P(x_n|class)
	      }
	      // confidence calced for given class
	      if(confidence > maxClass._2 || !initMax){ // found new prospective class
	        maxClass = Tuple2(cls, confidence)
	        initMax = true
	      }
	    }
	    result += Tuple3(actualClass, maxClass._2, maxClass._1)
    }
    result
  }
}