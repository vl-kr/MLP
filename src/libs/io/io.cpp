#include "io.h"
#include <iterator>

using namespace std;


Matrix loadFromCSV(string inFilePath) { // loads data form a .csv file into a Matrix
	ifstream inFile(inFilePath);
	cout << "Attempting to open file: " << inFilePath << endl;
	if (!inFile.is_open()) {
		cout << "File could not be opened" << endl;
		exit(1); // TODO: change this to throw an exception
	}

	string line;
	size_t numLines = 0;
	size_t numCols = 0;

	// Get the number of lines and columns in the file
	getline(inFile, line);
	numCols = count(line.begin(), line.end(), ',') + 1;
	numLines = count(istreambuf_iterator<char>(inFile), istreambuf_iterator<char>(), '\n') + 1;

	inFile.clear();  // Clear the end-of-file flag
	inFile.seekg(0);  // Reset file pointer to the beginning

	vector<vector<double>> data(numLines, vector<double>(numCols));

	int row = 0;
	while (getline(inFile, line)) {
		size_t pos = 0;
		int col = 0;
		while (pos < line.length()) {
			size_t commaPos = line.find(',', pos);
			string tmpStr;
			if (commaPos != string::npos) {
				tmpStr = line.substr(pos, commaPos - pos);
				pos = commaPos + 1;
			}
			else {
				tmpStr = line.substr(pos);
				pos = line.length();
			}
			data[row][col] = stod(tmpStr);
			col++;
		}
		row++;
	}
	return Matrix(data);
}

void writeToFile(string outFilePath, vector<double> outVect) { // used to create actualPredictions file
	ofstream inFile(outFilePath);
	assert(inFile.is_open());
	string line;
	for(double num : outVect){
		inFile << num << endl;
	}
}