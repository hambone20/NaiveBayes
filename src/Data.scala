package data
import io.Source
import collection.mutable.ArrayBuffer

object Data {
    val delims = Array(' ', ',', '.')
    
	def getData(file: String): Array[Array[Int]]= {
	  val dataLines = Source.fromFile(file).getLines
	  
	  var data = ArrayBuffer[Array[Int]]()
	  
	  for(elem <- dataLines){
	    data += elem.split(',').map(x => x.toInt).toArray
	  }
	  data.toArray
	}
}