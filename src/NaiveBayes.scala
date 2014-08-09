package naivebayes
import Math.log
import collection.mutable.ArrayBuffer

class NaiveBayes(trainingSet: Array[Array[Int]], classes: Array[Int], binning: Boolean = false) {
  // Mapping will contain P on class 
  
  val trainingSize = trainingSet.size
  val numAttributes = trainingSet(0).size - 1
  val attributeValues = if(!binning) Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16) 
                        else Array(Array(0, 1, 2, 3, 4), Array(5, 6, 7, 8), Array(9, 10, 11, 12), Array(13, 14, 15, 16)) //bins

  val numValues = attributeValues.size
  
  var model = collection.mutable.Map[String, Double]() // hold our model
  
  // A. compute prior probability for each class ie. P('1')
  computePForClasses()
  // B. compute conditional probabilities for each digits and for each value of each attribute. 
  // ie. P(Attribute1=0|'1') is the probability of that Attribute 1 has value 0, given that the class is digit '1' 
  // C. smooth the conditional probabilities using Laplace (add-one) smoothing
  for(cls <- classes){
    	computePForEachAttribute(cls) // calc for given class
  }
  
  def computePForClasses() {
    for(cls <- classes){
      val count = trainingSet.filter(x => x.last == cls).size
      model += Pair(cls.toString, count.toDouble / trainingSize)
    }
  }
  
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
                 
        model += Pair(key, (attValData.size + 1.0) / (clsData.size + numValues))
       }    
      }
  }
  
  def classify(testSet: Array[Array[Int]]) = {
    // calculate confidence interval for all classes, find class that maximizes
    var result = ArrayBuffer[(Int, Double, Int)]()
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
	        maxClass = Pair(cls, confidence)
	        initMax = true
	      }
	    }
	    result += Triple(actualClass, maxClass._2, maxClass._1)
    }
    result
  }
}