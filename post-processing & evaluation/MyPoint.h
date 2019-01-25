//
//  MyPoint.hpp
//  OpenCV_Project
//
//  Created by Alex Xu on 11/7/18.
//  Copyright © 2018 Alex Xu. All rights reserved.
//

#pragma once
#include <iostream>
#include <vector>

class MyPoint
{
public:
	MyPoint();
	~MyPoint();
	MyPoint(int id, int x, int y);
	MyPoint(int id, int x, int y, int firstNeighborId);
	void setId(int id);
	int getId();
	void setPosition(int x, int y);
	int getPositionX();
	int getPositionY();
	void addNeighbor(int neighborId);
	//void addNeighborVec(std::vector<int> neighborIdVec);
	std::vector<int> getNeighbors();
	void removeNeighbor(int idToRemove);
private:
	int id;
	int position_x;
	int position_y;
	std::vector<int> neighbors; // id of neighbors
};

