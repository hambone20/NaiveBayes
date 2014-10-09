package data
import io.Source

/** Singleton class that just handles data extraction
 */
object Data {
  val delims = Array(' ', ',', '.')

  /** Method to process data
   *  
   * The data file has 64 attributes, delimited by commas, with values between 0-16 
   * representing pixels in an image of a digit. The last element is the actual
   * class of the digit
   * @param fileName the filename to be processed
   * @return an array of arrays of integers representing the bitmap 
   *         image of a digit, with the final element being the actual digit
   */
  def getData(fileName: String): Array[Array[Int]] = {
    val dataLines = Source.fromFile(fileName).getLines    
    val data= for (elem <- dataLines) yield elem.split(delims).map(_.toInt)
    data.toArray
  }
}