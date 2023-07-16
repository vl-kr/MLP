#pragma once

#include "../matrix/matrix.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream> 

Matrix loadFromCSV(std::string inFilePath);
void writeToFile(std::string outFilePath, std::vector<double> outVect);