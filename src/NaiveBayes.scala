package naivebayes
//import Math.log
import scala.collection.mutable

/**
 * Workhorse of our Bayesian classifier
 *
 * @constructor creates a Bayesian classifier with a given training set
 *              and initializes the model
 * @param trainingSet the data to train our classifier on
 * @param classes the possible classfications
 * @param binning should the classes be binned
 */

class NaiveBayes(trainingSet: Array[Array[Int]], classes: Array[Int], binning: Boolean = false) {
  type Probability = Double
  val trainingSize = trainingSet.size
  val numAttributes = trainingSet.head.size - 1 // the last element is the actual classification
  val attributeValues = if (!binning) Range(0, 17).toArray
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
  for (cls <- classes) {
    computePForEachAttribute(cls) // calculate for all classes
  }
  /**
   * computes prior probability for all classes
   *
   * i.e. P('1')
   */
  private def computePForClasses() {
    for (cls <- classes) {
      val count = trainingSet.filter(x => x.last == cls).size
      val P: Probability = count.toDouble / trainingSize
      model += Tuple2(cls.toString, P)
    }
    assert(model.size == classes.size, "Error creating model for classes")
  }

  /**
   * computes prior probability for each attribute, adding Laplace smoothing
   *
   * i.e. P(Attribute1=0|'1') is inserted into the model with the key: 1=0|1
   * @param cls class to compute
   */
  private def computePForEachAttribute(cls: Int) {
    val clsData = trainingSet.filter(x => x.last == cls)

    for (attIdx <- 0 to numAttributes) {
      for (value <- attributeValues) {
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

  /**
   * classify based on the training data given
   *
   * calculate confidence interval for all classes, then find class that maximizes
   * @param testSet Data set to classify given the model
   * @return array of classifications where each element is a Tuple of (ActualClass, Confidence, Classification)
   */
  def classify(testSet: Array[Array[Int]]): Array[(ActualClass, ConfidenceLevel, ClassifiedClass)] = {
    var result = Array[(ActualClass, ConfidenceLevel, ClassifiedClass)]()
    for (test <- testSet) {
      val actualClass: ActualClass = test.last
      var maxClass: Option[(ClassifiedClass, ConfidenceLevel)] = None

      for (cls <- classes) {
        val key = cls.toString
        var confidence: ConfidenceLevel = Math.log(model(key)) // log P(class)
        for (attIdx <- 0 to numAttributes) {
          val value = attributeValues match {
            case arr: Array[Int] => test(attIdx).toString 
            case arr: Array[Array[Int]] => findAttribute(arr, test(attIdx))
            case _ => throw new Exception("Attributes were improperly formed")
          }
          val key = s"$attIdx=${value}|$cls"
          confidence += Math.log(model(key)) // + log(P(x_n|class)
        }
        maxClass = maxClass match {
          case None => Some(Tuple2(cls, confidence))
          case Some(Tuple2(_, maxConfidence)) if confidence > maxConfidence => Some(Tuple2(cls, confidence))
          case _ => maxClass
        }
      }
      val el: (ActualClass, ConfidenceLevel, ClassifiedClass) = maxClass match {
        case Some(Tuple2(maxClassified, maxConfidence)) => Tuple3(actualClass, maxConfidence, maxClassified)
        case None => throw new Exception("No confidence")
      }
      result = result :+ el
    }
    result
  }

  /**
   * A helper method for classify
   *
   * Finds attribute when binning is done
   * @param haystack array to search for target
   * @param needle target to be found
   * @return key value
   */
  private def findAttribute(haystack: Array[Array[Int]], needle: Int): String = {
    for (i <- 0 until haystack.size) {
      if (haystack(i).contains(needle)) {
        return haystack(i).mkString
      }
    }
    throw new Exception("Attribute not found")
  }
}