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

	static Matrix VectorToMatrix(const std::vector<double>& vector, bool transposeVector);

	void AddMatrix(const Matrix& MatrixB);

	static Matrix _AddMatrices(const Matrix& MatrixA, const Matrix& MatrixB); //use AddMatrix instead

	void MultiplyByScalar(int scalar);

	static Matrix _MultiplyMatrixByScalar(const Matrix& MatrixA, int scalar); //use MultiplyByScalar instead

	static Matrix _MultiplyMatrices(const Matrix& MatrixA, const Matrix& MatrixB); //use MultiplyMatricesParallel instead

	static Matrix MultiplyMatricesParallel(const Matrix& MatrixA, const Matrix& MatrixB);
	static Matrix MultiplyMatricesParallel(const Matrix& MatrixA, const std::vector<double>& vectorB, bool transposeVector=true);

	static std::vector<double> MultiplyMatrixByVector(const Matrix& MatrixA, const std::vector<double>& VectorB);

	static Matrix RandomMatrix(size_t maxRows, size_t maxCols, int maxVal);

	static Matrix RandomMatrixSetSize(size_t rows, size_t cols, double interval);

	static bool MatricesEqual(const Matrix& MatrixA, const Matrix& MatrixB);
};

// TODO: Reference additional headers your program requires here.
