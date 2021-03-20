#include "Neuron.h"

double Neuron::Init_Neuron() const noexcept
{
	//std::random_device rd;
 //   std::mt19937 mt(rd());
 //   std::uniform_real_distribution<double> dist(0.0, 1.0);

	//return dist(rd);

	return static_cast<double>(rand()/ RAND_MAX);
}
