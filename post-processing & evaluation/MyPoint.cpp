//
//  MyPoint.cpp
//  OpenCV_Project
//
//  Created by Alex Xu on 11/7/18.
//  Copyright © 2018 Alex Xu. All rights reserved.
//

// #include "pch.h"
#include "MyPoint.h"


MyPoint::~MyPoint()
{
}

MyPoint::MyPoint() {
	this->position_x = -1;
	this->position_y = -1;
	this->id = -1;
	std::vector<int> n;
	this->neighbors = n;
}

MyPoint::MyPoint(int id, int x, int y) {
	this->id = id;
	this->position_x = x;
	this->position_y = y;
	this->neighbors.clear();
}

MyPoint::MyPoint(int id, int x, int y, int firstNeighborId) {
	this->id = id;
	this->position_x = x;
	this->position_y = y;
	this->neighbors.push_back(firstNeighborId);
}

void MyPoint::setId(int id) {
	this->id = id;
}

int MyPoint::getId() {
	return this->id;
}

void MyPoint::setPosition(int x, int y) {
	this->position_x = x;
	this->position_y = y;
}

int MyPoint::getPositionX() {
	return this->position_x;
}

int MyPoint::getPositionY() {
	return this->position_y;
}

void MyPoint::addNeighbor(int neighborId) {
	bool foundSameId = false;
	for (int i = 0; i < this->neighbors.size(); i++) {
		if (this->neighbors[i] == neighborId) {
			foundSameId = true;
		}
	}
	if (!foundSameId) {
		this->neighbors.push_back(neighborId);
	}
}

//void MyPoint::addNeighborVec(std::vector<int> neighborIdVec) {
//	this->neighbors.insert(this->neighbors.begin(), neighborIdVec.begin(), neighborIdVec.end());
//	//sort(this->neighbors.begin(), this->neighbors.end());
//	std::sort
//	this->neighbors.erase(unique(this->neighbors.begin(), this->neighbors.end()), this->neighbors.end());
//}

std::vector<int> MyPoint::getNeighbors() {
	return this->neighbors;
}

void MyPoint::removeNeighbor(int idToRemove) {
	int index = -1;
	for (int i = 0; i < this->neighbors.size(); i++) {
		if (idToRemove == neighbors[i]) {
			index = i;
			break;
		}
	}
	if (index == -1) {
		std::cout << "Remove Failed! Neighbor ID Not Found!" << std::endl;
		//exit(-1);
	} else {
		this->neighbors.erase(neighbors.begin() + index);
	}
	
}
