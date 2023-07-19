#include "matrix.h"

using namespace std;

double uniformRandom() {
	/*
	random number generator with uniform distribution, interval (0 to 1)
	*/
	double randNum = ((double)rand()) / (RAND_MAX); // generates number in interval <0 to 1>
	if (randNum == 0 || randNum == 1) { // if number is 0 or 1, move the number slightly towards 0.5
		randNum = nextafter(randNum, 0.5); // returns the next representable value of randNum in the direction of 0.5
	}
	return randNum;
}

double normalRandom() {
	/*
	random number generator with normal distribution (mean 0, std 1)
	*/
	double rand1 = uniformRandom();
	double rand2 = uniformRandom();
	double pi = 3.14159265358979323846;
	return cos(2 * pi * rand1) * sqrt(-2.0 * log(rand2)); // Box–Muller transform
}

Matrix::Matrix(vector<vector<double>> data) {
	assert(data.size() > 0);
	this->data = move(data);
	this->rows = this->data.size();
	this->cols = this->data[0].size();
}

void Matrix::PrintMatrix() {
	for (vector<double> innerVector : this->data)
	{
		for (int number : innerVector)
		{
			cout << number << ' ';
		}
		cout << endl;
	}
	cout << endl;
}

Matrix Matrix::Transpose(const Matrix& matrix, bool dropBiases) {
	/*
	return new transposed matrix

	dropBiases 	:	if true, the last column of the original Matrix is ignored
	*/
	int cols = matrix.cols - (int)dropBiases;
	Matrix newMatrix(vector<vector<double>>(cols, vector<double>(matrix.rows)));
	int row;
#pragma omp parallel for
for (row = 0; row < matrix.rows; row++) {
		for (int col = 0; col < cols; col++) {
			newMatrix.data[col][row] = matrix.data[row][col];
		}
	}
	return newMatrix;
}

void Matrix::AddMatrix(const Matrix& MatrixB) {
	/*
	adds MatrixB to this matrix, modifies this matrix,
	MatrixB must have the same dimensions as this matrix
	*/
	if (this->cols != MatrixB.cols || this->rows != MatrixB.rows) {
		throw invalid_argument("Matrix dimensions must match");
	}
	int row;
#pragma omp parallel for
	for (row = 0; row < this->rows; row++) {
		for (size_t col = 0; col < this->cols; col++) {
			this->data[row][col] += MatrixB.data[row][col];
		}
	}
}

Matrix Matrix::MultiplyMatricesParallel(const Matrix& MatrixA, const Matrix& MatrixB) {
	/*
	multiplies MatrixA by MatrixB, returns new matrix
	MatrixA.cols must be equal to MatrixB.rows

	MatrixA							:	matrix to be multiplied by MatrixB
	MatrixB							:	matrix multiplying MatrixA
	*/
	if (MatrixA.cols != MatrixB.rows) {
		throw invalid_argument("The matrix dimensions are not compatible for multiplication");
	}
	vector<vector<double>> data(MatrixA.rows, vector<double>(MatrixB.cols));
	int rowA;
#pragma omp parallel for
	for (rowA = 0; rowA < MatrixA.rows; rowA++) {
		for (size_t colA_rowB = 0; colA_rowB < MatrixA.cols; colA_rowB++) {
			for (size_t colB = 0; colB < MatrixB.cols; colB++) {
				data[rowA][colB] += MatrixA.data[rowA][colA_rowB] * MatrixB.data[colA_rowB][colB];
			}
		}
	}
	return Matrix(data);
}


Matrix Matrix::VectorToMatrix(const vector<double>& vect, bool transposeVector) {
	/*
	convers vector to matrix and returns the resulting matrix
	
	vect							:	vector to be converted to matrix
	transposeVector					:	whethr to transpose the vector first, default is true
	*/
	if (!transposeVector) {
		return Matrix(vector<vector<double>>(1, vect));
	}
	vector<vector<double>> data(vect.size(), vector<double>(1));
	for (size_t i = 0; i < vect.size(); i++) {
		data[i][0] = vect[i];
	}
	return Matrix(data);
}

vector<double> Matrix::MultiplyMatrixByVector(const Matrix& MatrixA, const vector<double>& VectorB, bool addBias) {
	/*
	multiplies matrix by vector and returns the result as a vector

	MatrixA							:	matrix to be multiplied by vector
	VectorB							:	vector multiplying matrix
	addBias							:	whether to add bias to the result, default is false		!!!IMPORTANT!!! if true, MatrixA.cols must be equal to VectorB.size() + 1
	*/
	if (!addBias && MatrixA.cols != VectorB.size()) {
		throw invalid_argument("The matrix cannot be multiplied by vector of this size");
	}
	else if (addBias && MatrixA.cols != VectorB.size() + 1) {
		throw invalid_argument("The matrix with bias cannot be multiplied by vector of this size");
	}
	vector<double> data(MatrixA.rows);
	int rowA;
#pragma omp parallel for
	for (rowA = 0; rowA < MatrixA.rows; rowA++) {
		for (size_t colA_rowB = 0; colA_rowB < VectorB.size(); colA_rowB++) {
			data[rowA] += MatrixA.data[rowA][colA_rowB] * VectorB[colA_rowB];
		}
		if (addBias) {
			data[rowA] += MatrixA.data[rowA][VectorB.size()];
		}
	}
	return data;
}

Matrix Matrix::MultiplyVectors(const vector<double>& VectorA, const vector<double>& VectorB, bool addBias) {
	/*
	multiplies 2 vectors and returns the result as a matrix

	VectorA							:	vector to be multiplied
	VectorB							:	vector multiplying
	addBias							:	whether to extend VectorB by 1 (due to bias), default is false
	*/
	int cols = VectorB.size() + (int)addBias;
	Matrix newMatrix = Matrix(vector<vector<double>>(VectorA.size(), vector<double>(cols)));
	int row;
#pragma omp parallel for
	for (row = 0; row < VectorA.size(); row++) {
		for (size_t col = 0; col < VectorB.size(); col++) {
			newMatrix.data[row][col] = VectorA[row] * VectorB[col];
		}
		if (addBias) {
			newMatrix.data[row][VectorB.size()] = VectorA[row];
		}
	}
	return newMatrix;
}

Matrix Matrix::RandomMatrixSetSize(size_t rows, size_t cols, double variance) {
	/*
	creates a new matrix of given dimenstions and fills it with random numbers from normal distribution with given variance
	used for weights initialization
	*/
	vector<vector<double>> data;
	for (size_t i = 0; i < rows; i++) {
		vector<double> tmp;
		for (size_t j = 0; j < cols; j++) {
			double randNum = normalRandom();
			tmp.push_back(randNum * variance);
		}
		data.push_back(tmp);
	}
	return Matrix(data);
}
