// Most of the Matrix operation functions are not actually used anywhere

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

Matrix::Matrix() {
	cols = 0;
	rows = 0;
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

Matrix Matrix::Transpose() {
	Matrix newMatrix = Matrix();
	for (size_t col = 0; col < this->cols; col++) {
		vector<double> tempVCT;
		for (size_t row = 0; row < this->rows; row++) {
			tempVCT.push_back(this->data[row][col]);
		}
		newMatrix.data.push_back(tempVCT);
	}
	newMatrix.cols = this->rows;
	newMatrix.rows = this->cols;
	return newMatrix;
}

void Matrix::AddMatrix(const Matrix& MatrixB) {
	assert(this->cols == MatrixB.cols && this->rows == MatrixB.rows);
//#pragma omp parallel for
	for (size_t row = 0; row < this->rows; row++) {
		for (size_t col = 0; col < this->cols; col++) {
			this->data[row][col] += MatrixB.data[row][col];
		}
	}
}

Matrix Matrix::_AddMatrices(const Matrix& MatrixA, const Matrix& MatrixB) {
	Matrix newMatrix = Matrix();
	assert(MatrixA.cols == MatrixB.cols && MatrixA.rows == MatrixB.rows);
	for (size_t row = 0; row < MatrixA.rows; row++) {
		vector<double> tempVCT;
		for (size_t col = 0; col < MatrixA.cols; col++) {
			tempVCT.push_back(MatrixA.data[row][col] + MatrixB.data[row][col]);
		}
		newMatrix.data.push_back(tempVCT);
	}
	newMatrix.rows = MatrixA.rows;
	newMatrix.cols = MatrixA.cols;
	return newMatrix;
}

void Matrix::MultiplyByScalar(int scalar) {
//#pragma omp parallel for
	for (size_t row = 0; row < this->rows; row++) {
		for (size_t col = 0; col < this->cols; col++) {
			this->data[row][col] *= scalar;
		}
	}
}

Matrix Matrix::_MultiplyMatrixByScalar(const Matrix& MatrixA, int scalar) {
	vector<vector<double>> data(MatrixA.rows, vector<double>(MatrixA.cols));
	Matrix newMatrix = Matrix();
	for (size_t row = 0; row < MatrixA.rows; row++) {
		vector<double> tempVCT;
		for (size_t col = 0; col < MatrixA.cols; col++) {
			tempVCT.push_back(MatrixA.data[row][col] * scalar);
		}
		newMatrix.data.push_back(tempVCT);
	}
	newMatrix.rows = MatrixA.rows;
	newMatrix.cols = MatrixA.cols;
	return newMatrix;
}

Matrix Matrix::_MultiplyMatrices(const Matrix& MatrixA, const Matrix& MatrixB) {
	Matrix newMatrix = Matrix();
	assert(MatrixA.cols == MatrixB.rows);
	for (vector<double> rowA : MatrixA.data) {
		vector<double> tempVCT;
		for (size_t i = 0; i < MatrixB.cols; i++) {
			int sum = 0;
			for (size_t j = 0; j < MatrixA.cols; j++) {
				sum += rowA[j] * MatrixB.data[j][i];
			}
			tempVCT.push_back(sum);
		}
		newMatrix.data.push_back(tempVCT);
	}
	newMatrix.rows = MatrixA.rows;
	newMatrix.cols = MatrixB.cols;
	return newMatrix;
}


Matrix Matrix::MultiplyMatricesParallel(const Matrix& MatrixA, const Matrix& MatrixB) {
	/*
	multiplies MatrixA by MatrixB, returns new matrix
	MatrixA.cols must be equal to MatrixB.rows

	MatrixA							:	matrix to be multiplied by MatrixB
	MatrixB							:	matrix multiplying MatrixA
	*/
	assert(MatrixA.cols == MatrixB.rows);
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
	addBias							:	whether to add bias to the result, default is false		!!!IMPORTANT!!! if true, VectorB.size() must be equal to MatrixA.cols + 1
	*/
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

Matrix Matrix::RandomMatrix(size_t maxRows, size_t maxCols, int maxVal) {
	size_t rows = rand() % maxRows + 1;
	size_t cols = rand() % maxCols + 1;
	vector<vector<double>> d;
	for (size_t i = 0; i < rows; i++) {
		vector<double> tmp;
		for (size_t j = 0; j < cols; j++) {
			tmp.push_back(rand() % maxVal);
		}
		d.push_back(tmp);
	}
	return Matrix(d);
}

Matrix Matrix::RandomMatrixSetSize(size_t rows, size_t cols, double variance) { // used for weight initialization
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

bool Matrix::MatricesEqual(const Matrix& MatrixA, const Matrix& MatrixB) { // just for testing
	assert(MatrixA.cols == MatrixB.cols && MatrixA.rows == MatrixB.rows);
	for (size_t row = 0; row < MatrixA.rows; row++) {
		for (size_t col = 0; col < MatrixA.cols; col++) {
			if (MatrixA.data[row][col] != MatrixB.data[row][col]) {
				return false;
			}
		}
	}
	return true;
}