// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "Json.h"
#include "JsonVariables.h"
#include "Engine.h"
#include "Templates/SharedPointer.h"
#include "Misc/FileHelper.h"
#include "Serialization/JsonSerializer.h"
#include "Dom/JsonObject.h"
#include "JsonUtilities/Public/JsonObjectConverter.h"
#include "CoreMinimal.h"

/**
 * 
 */
class AI_CAR_API NeuralNetwork
{
public:
	NeuralNetwork();
	NeuralNetwork(FString);
	NeuralNetwork(TArray<int> topology);
	~NeuralNetwork();

	void Init(TArray<int> topology);

	TArray<float> forward(TArray<float> data);

	TArray<TArray<TArray<float>>> NN;

private:

	float Sigmoid(float x) { return -1 + 2 / (1 + FGenericPlatformMath::Exp(-x)); }
};

