#ifndef _LAYER_H_INCLUDED_
#define _LAYER_H_INCLUDED_

#include <random>
#include <vector>

#include "Neuron.h"

class Layer
{
protected:
	std::vector<Neuron> List_Of_Neurons;
	std::size_t numberofNeuronInLayer;

public:
	Layer() = delete;

	void Print_Layer() const noexcept;
	
	~Layer() = default;
};


#endif /* _LAYER_H_INCLUDED_ */