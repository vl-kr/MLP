#pragma once

#include "../matrix/matrix.h"
#include <iostream>
#include <sstream> 
#include <fstream>
#include <string>
#include <algorithm>

Matrix loadFromCSV(std::string inFilePath, int normalizationDivisor = 1);
void writeToFile(std::string outFilePath, std::vector<double> outVect);