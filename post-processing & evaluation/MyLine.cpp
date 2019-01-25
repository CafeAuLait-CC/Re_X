//
//  MyLine.cpp
//  OpenCV_Project
//
//  Created by Alex Xu on 11/7/18.
//  Copyright © 2018 Alex Xu. All rights reserved.
//

#include "pch.h"
#include "MyLine.h"


MyLine::MyLine()
{
}


MyLine::~MyLine()
{
}

MyLine::MyLine(MyPoint p1, MyPoint p2) {
	if (p1.getId() < p2.getId()) {
		this->startPoint = p1;
		this->endPoint = p2;
	}
	else {
		this->startPoint = p2;
		this->endPoint = p1;
	}
}

MyLine::MyLine(MyPoint p1, MyPoint p2, int id) {
	if (p1.getId() < p2.getId()) {
		this->startPoint = p1;
		this->endPoint = p2;
	}
	else {
		this->startPoint = p2;
		this->endPoint = p1;
	}
	this->lineId = id;
}

std::vector<MyPoint> MyLine::getPointsOnLine() {
	std::vector<MyPoint> points;
	points.push_back(this->startPoint);
	points.push_back(this->endPoint);
	return points;
}

void MyLine::setTwoPointsForLine(MyPoint p1, MyPoint p2) {
	if (p1.getId() < p2.getId()) {
		this->startPoint = p1;
		this->endPoint = p2;
	}
	else {
		this->startPoint = p2;
		this->endPoint = p1;
	}
}
