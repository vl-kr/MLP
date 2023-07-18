// matrix.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include <omp.h>
#include <cmath>


class Matrix
{
public:

	size_t rows;
	size_t cols;

	std::vector<std::vector<double>> data;

	Matrix();

	Matrix(std::vector<std::vector<double>> data);

	void PrintMatrix();

	Matrix Transpose();

	static Matrix VectorToMatrix(std::vector<double>& vector, bool transposeVector);

	void AddMatrix(Matrix MatrixB);

	static Matrix _AddMatrices(Matrix& MatrixA, Matrix& MatrixB); //use AddMatrix instead

	void MultiplyByScalar(int scalar);

	static Matrix _MultiplyMatrixByScalar(Matrix& MatrixA, int scalar); //use MultiplyByScalar instead

	static Matrix _MultiplyMatrices(Matrix& MatrixA, Matrix& MatrixB); //use MultiplyMatricesParallel instead

	static Matrix MultiplyMatricesParallel(Matrix& MatrixA, Matrix& MatrixB);
	static Matrix MultiplyMatricesParallel(Matrix& MatrixA, std::vector<double>& vectorB, bool transposeVector=true);

	static Matrix RandomMatrix(size_t maxRows, size_t maxCols, int maxVal);

	static Matrix RandomMatrixSetSize(size_t rows, size_t cols, double interval);

	static bool MatricesEqual(Matrix& MatrixA, Matrix& MatrixB);
};

// TODO: Reference additional headers your program requires here.
