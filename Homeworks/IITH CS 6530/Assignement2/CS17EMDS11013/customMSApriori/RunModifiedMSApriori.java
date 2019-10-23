package customMSApriori;

import java.io.IOException;

/**
* Run the AlgoMSApriori algorithm by passing all necessary values
*/
public class RunModifiedMSApriori {
	public static void main(String [] arg) throws IOException{
		AlgoMSApriori msApriori = new AlgoMSApriori();
		String inputFile = "dataset/retail1/retail1.txt";
		String outputFile = "output_spmf.txt";
		double beta = 0.5; // beta value to calcualte MIS
		double LS = 0.01;
		double SPD = 0.05;
		String canNotbeTogetherFile = "can_not_be_together.txt";
		String mustBeTogetherFile = "must_be_together.txt";
		msApriori.runAlgorithm(inputFile, outputFile, beta, LS, SPD, canNotbeTogetherFile, mustBeTogetherFile);
		msApriori.printStats();
	}
}

/**
To run
javac customMSApriori\*.java
java -cp . customMSApriori.RunModifiedMSApriori
*/
