#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include <omp.h>
#include <math.h>

class Matrix
{
public:

	size_t rows;
	size_t cols;

	std::vector<std::vector<double>> data;

	Matrix();

	Matrix(std::vector<std::vector<double>> data);

	void PrintMatrix();

	static Matrix Transpose(const Matrix& Matrix, bool dropBiases = false);

	static Matrix VectorToMatrix(const std::vector<double>& vector, bool transposeVector);

	void AddMatrix(const Matrix& MatrixB);

	static Matrix MultiplyMatricesParallel(const Matrix& MatrixA, const Matrix& MatrixB);

	static std::vector<double> MultiplyMatrixByVector(const Matrix& MatrixA, const std::vector<double>& VectorB, bool addBias = false);

	static Matrix MultiplyVectors(const std::vector<double>& VectorA, const std::vector<double>& VectorB, bool addBias = false);

	static Matrix RandomMatrixSetSize(size_t rows, size_t cols, double interval);
};
