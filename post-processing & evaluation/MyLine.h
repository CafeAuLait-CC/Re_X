//
//  MyLine.hpp
//  OpenCV_Project
//
//  Created by Alex Xu on 11/7/18.
//  Copyright © 2018 Alex Xu. All rights reserved.
//

#pragma once
#include <iostream>
#include "MyPoint.h"

class MyLine
{
public:
	MyLine();
	~MyLine();
	MyLine(MyPoint p1, MyPoint p2);
	MyLine(MyPoint p1, MyPoint p2, int id);
	std::vector<MyPoint> getPointsOnLine();
	void setTwoPointsForLine(MyPoint p1, MyPoint p2);

private:
	int lineId;
	MyPoint startPoint;
	MyPoint endPoint;
};

