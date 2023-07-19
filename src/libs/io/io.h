#pragma once

#include "../matrix/matrix.h"
#include <fstream>
#include <string>

Matrix loadFromCSV(std::string inFilePath, int normalizationDivisor = 1);
void writeToFile(std::string outFilePath, std::vector<double> outVect);