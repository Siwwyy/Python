#ifndef _NEURON_H_INCLUDED_
#define _NEURON_H_INCLUDED_

#include <random>
#include <vector>


class Neuron
{
protected:
	std::vector<double> list_of_weights_in;
	std::vector<double> list_of_weights_out;
	double output_Value;
	double error;
	double sensibility;
public:
	Neuron() = delete;


	double Init_Neuron() const noexcept;

	
	~Neuron() = default;
};


#endif /* _NEURON_H_INCLUDED_ */