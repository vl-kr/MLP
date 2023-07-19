#include "util.h"

using namespace std;

int vecToScalar(vector<double> classVector) {
	int maxIndex = 0;
	for (size_t i = 0; i < classVector.size(); i++)
	{
		if (classVector[i] > classVector[maxIndex])
			maxIndex = i;
	}
	return maxIndex;
}

vector<double> scalarToVec(int scalar) {
	vector<double> res(10);
	res[scalar] = 1;
	return res;
}