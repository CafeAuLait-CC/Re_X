#include <iostream>
#include <fstream>
#include <time.h>

#ifdef __APPLE__
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#endif

#ifdef linux
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#endif

#ifdef _WIN32
#include <io.h>
#include <direct.h>
#endif

#include <opencv2/opencv.hpp>
#include "MyLine.h"

#define BASE_PATH "../results/"
#define P2P_THRESHOLD 30
#define P2L_THRESHOLD 30

#define PRE_MAX_PATCH_LOC_X 81  // default value for MAX_PATCH_LOC
#define PRE_MAX_PATCH_LOC_Y 81
#define IMAGE_TILE_SIZE_ROWS 8192   // default value for IMAGE_TILE_SIZE
#define IMAGE_TILE_SIZE_COLS 8192

using namespace std;
using namespace cv;

void startEval(vector<string> cities);
void generateErrorImage(vector<string> cities);
void evaluateError(vector<string> cities);
bool searchAround(int rowIdx, int colIdx, Mat templateImg);

void generateAllPatches(vector<string> cities, string INPUT_PATH, string OUTPUT_PATH);

void cleanUpHoughLineImage(string cityName, string MODEL_NAME, string INPUT_PATH, string OUTPUT_PATH, Size IMAGE_TILE_SIZE);
float point2PointDistance(MyPoint p1, MyPoint p2);
Point getIntersectionOfTwoLines(float k, float b, MyLine l);
vector<vector<Vec4i>> houghLineOnPatch(string cityName, string MODEL_NAME, string INPUT_PATH, Size IMAGE_TILE_SIZE, int MAX_PATCH_LOC_X, int MAX_PATCH_LOC_Y);
float point2LineDistance(Point p, Vec4i line);
bool isParallelLine(Vec4i l1, Vec4i l2);

struct ID4DeletedPoint {
    int deletedID;
    int newID;
};

MyPoint getMyPointWithId(int pointId, vector<MyPoint> allPointsVec, vector<ID4DeletedPoint> deletedIDVec);
int getIndexOfMyPointWithId(int pointId, vector<MyPoint> allPointsVec, vector<ID4DeletedPoint> deletedIDVec);

void parseArgs(int argc, const char * argv[], int &mode, string &MODEL_NAME, string &INPUT_PATH, string &OUTPUT_PATH, Size &IMAGE_TILE_SIZE, int &MAX_PATCH_LOC_X, int &MAX_PATCH_LOC_Y);
void createFolder(string folderPath);


// TODO:
// 1. Change to .tif file
// 2. read image file name from file


int main(int argc, const char * argv[]) {

    int mode;
    string MODEL_NAME;
    string INPUT_PATH;
    string OUTPUT_PATH;
    Size IMAGE_TILE_SIZE;
    int MAX_PATCH_LOC_X;
    int MAX_PATCH_LOC_Y;
    parseArgs(argc, argv, mode, MODEL_NAME, INPUT_PATH, OUTPUT_PATH, IMAGE_TILE_SIZE, MAX_PATCH_LOC_X, MAX_PATCH_LOC_Y);

    // To test the parser
//    cout << mode << endl;
//    cout << MODEL_NAME << endl;
//    cout << INPUT_PATH << endl;
//    cout << OUTPUT_PATH << endl;
//    cout << IMAGE_TILE_SIZE << endl;
//    cout << MAX_PATCH_LOC_X << endl;
//    cout << MAX_PATCH_LOC_Y << endl;
    
    vector<string> cities;
    cities.push_back("amsterdam");
    cities.push_back("boston");
    cities.push_back("chicago");
    cities.push_back("denver");
    cities.push_back("kansas city");
    cities.push_back("la");
    cities.push_back("montreal");
    cities.push_back("new york");
    cities.push_back("paris");
    cities.push_back("pittsburgh");
    cities.push_back("saltlakecity");
    cities.push_back("san diego");
    cities.push_back("tokyo");
    cities.push_back("toronto");
    cities.push_back("vancouver");
    
    switch (mode) {
        case 0:
            generateAllPatches(cities, MODEL_NAME, INPUT_PATH, OUTPUT_PATH);   // Gerenate patches for testing
            break;
        case 1:
            cleanUpHoughLineImage(cityName, MODEL_NAME, INPUT_PATH, OUTPUT_PATH, IMAGE_TILE_SIZE);  // Iterative Hough Transform on patches
            break;
        default:
//            startEval(cities);        // Evaluate results (IoU and F1 score)
//            drawDiffMapOnRGB(cities);     // Draw TP, FP, FN on RGB image
            break;
    }
    
    return 0;
    
}

void parseArgs(int argc, const char * argv[], int &mode, string &MODEL_NAME, string &INPUT_PATH, string &OUTPUT_PATH, Size &IMAGE_TILE_SIZE, int &MAX_PATCH_LOC_X, int &MAX_PATCH_LOC_Y) {
    if (argc < 4) {
        cout << "\nUsage: ./Re_X mode -n model_name [...opts]" << endl;
        cout << "\n    mode:  0: generateAllPatches()\n\t   1: cleanUpHoughLineImage()\n\t   2: startEval() & drawDiffMapOnRGB()" << endl;
        cout << "\n    -n:  the folder name used to save the trained model." << endl;
        cout << "\n    opts:  -w --image_width\t\t(default 8192)\n\t   -h --image_height\t\t(default 8192)\n\t   -c --patch_cols\t\t(default 81 -- file name from 0 to 80)\n\t   -r --patch_rows\t\t(default 81 -- file name from 0 to 80)\n\t   -i --input_folder\t\t(leave empty to use default setting)\n\t   -o --output_folder\t\t(leave empty to use default setting)" << endl;
        exit(-1);
    }
    
    /** Select program running mode.
        0: generateAllPatches()
        1: cleanUpHoughLineImage()
        2: startEval() & drawDiffMapOnRGB()
    */
    mode = stoi(argv[1]);
    if (mode < 0 || mode > 2) {
        cout << "Wrong mode number!" << endl;
        exit(-1);
    }
    
    bool n_set = false;
    bool w_set = false;
    bool h_set = false;
    bool c_set = false;
    bool r_set = false;
    
    if (argc % 2 != 0) {
        cout << "Wrong argument number!" << endl;
        exit(-1);
    }
    for (int i = 2; i < argc; i += 2) {
        if (!strcmp(argv[i], "-i") || !strcmp(argv[i], "--input_folder")) {
            INPUT_PATH = argv[i+1];
            if (access(argv[i+1], 0) == -1) {  // test if it's correct or not
                cout << argv[i] << endl;
                cout << "The input directory doesn't exist!" << endl;
                exit(-1);
            }
            if (INPUT_PATH[INPUT_PATH.size()-1] != '/') {   // append '/' to the end
                INPUT_PATH = INPUT_PATH + '/';
            }
        } else if (!strcmp(argv[i], "-o") || !strcmp(argv[i], "--output_folder")) {
            OUTPUT_PATH = argv[i+1];
            if (OUTPUT_PATH[OUTPUT_PATH.size()-1] != '/') {
                OUTPUT_PATH = OUTPUT_PATH + '/';
            }
        } else if (!strcmp(argv[i], "-n")) {
            MODEL_NAME = argv[i+1];
            if (access(("../results/" + MODEL_NAME).c_str(), 0) == 0) {
                n_set = true;
            }
        } else {
            char arg;
            if (strlen(argv[i]) == 2) {
                arg = argv[i][1];
            } else if (strlen(argv[i]) > 9) {
                arg = argv[i][8];
            } else {
                cout << "Wrong argument name: " << argv[i] << endl;
                exit(-1);
            }
            try {
                int arg_val = stoi(argv[i+1]);
                if (arg_val < 1) {
                    throw arg_val;
                }
                switch (arg) {
                    case 'w':
                        IMAGE_TILE_SIZE.width = arg_val;
                        w_set = true;
                        break;
                    case 'h':
                        IMAGE_TILE_SIZE.height = arg_val;
                        h_set = true;
                        break;
                    case 'c':
                        MAX_PATCH_LOC_X = arg_val;
                        c_set = true;
                        break;
                    case 'r':
                        MAX_PATCH_LOC_Y = arg_val;
                        r_set = true;
                        break;
                    default:
                        cout << "Wrong argument name: " << argv[i] << endl;
                        exit(-1);
                };
            } catch (...) {
                cout << "Invalid argument value: " << argv[i] << ": " << argv[i+1] << endl;
                exit(-1);
            }
        }
    }
    
    if (!n_set) {
        cout << "Model name is incorrect: " << MODEL_NAME << endl;
        exit(-1);
    }
    // If not set by arguments, use default values
    if (!w_set) {
        IMAGE_TILE_SIZE.width = IMAGE_TILE_SIZE_COLS;
    }
    if (!h_set) {
        IMAGE_TILE_SIZE.height = IMAGE_TILE_SIZE_ROWS;
    }
    if (!c_set) {
        MAX_PATCH_LOC_X = PRE_MAX_PATCH_LOC_X;
    }
    if (!r_set) {
        MAX_PATCH_LOC_Y = PRE_MAX_PATCH_LOC_Y;
    }
}

void createFolder(string folderPath) {
#ifdef __APPLE__
    if (mkdir(folderPath.c_str(), 0744) == -1) {
        cout << "Failed to create folder: " << folderPath << endl;
        exit(-1);
    }
#endif
    
#ifdef linux
    if (mkdir(folderPath.c_str(), 0744) == -1) {
        cout << "Failed to create folder: " << folderPath << endl;
        exit(-1);
    }
#endif
    
#ifdef _WIN32
    if (mkdir(const char *_Path) == -1) {
        cout << "Failed to create folder: " << folderPath << endl;
        exit(-1);
    }
#endif
}

void cleanUpHoughLineImage(string cityName, string MODEL_NAME, string INPUT_PATH, string OUTPUT_PATH, Size IMAGE_TILE_SIZE, int MAX_PATCH_LOC_X, int MAX_PATCH_LOC_Y) {
    
    time_t startTime = time(NULL);
    
    cout << "# Processing " << cityName << " ..." << endl;
    cout << "1. Detecting Hough Lines..." << endl;
    vector<vector<Vec4i>> allPatchesOfLines = houghLineOnPatch(cityName, MODEL_NAME, INPUT_PATH, IMAGE_TILE_SIZE, MAX_PATCH_LOC_X, MAX_PATCH_LOC_Y);    // allLines.size() is the number of patches, allLine[i].size() is the number of lines in patch i
    cout << "   - Finished in " << time(NULL) - startTime << " seconds." << endl;
    cout << "2. Extracting Points & Lines..." << endl;
    time_t currentTime = time(NULL);
    vector<MyPoint> allRoadPoints;
    
    // Start building a vector to store all points
    int pointCounter = 0;
    for (int patch_num = 0; patch_num < allPatchesOfLines.size(); patch_num++) {
        for (int i = 0; i < allPatchesOfLines[patch_num].size(); i++) {
            MyPoint p1(pointCounter, allPatchesOfLines[patch_num][i][0], allPatchesOfLines[patch_num][i][1]);
            pointCounter++;
            MyPoint p2(pointCounter, allPatchesOfLines[patch_num][i][2], allPatchesOfLines[patch_num][i][3]);
            pointCounter++;
            p1.addNeighbor(p2.getId());
            p2.addNeighbor(p1.getId());
            allRoadPoints.push_back(p1);
            allRoadPoints.push_back(p2);
        }
    }
    
    // Start building a vector to store all lines
    vector<MyLine> allRoadLines;
    vector<ID4DeletedPoint> deletedIDVec;   // map from ID of deleted points to ID of new points. for step 4
    // since I didn't change the connect relations (neighbors) while removing duplicated points
    int lineId = 0;
    allRoadLines.push_back(MyLine(allRoadPoints[0], getMyPointWithId(allRoadPoints[0].getNeighbors()[0], allRoadPoints, deletedIDVec), lineId));
    lineId++;
    for (int i = 0; i < allRoadPoints.size(); i++) {
        bool foundSameLine = false;
        for (int j = 0; j < allRoadLines.size(); j++) {
            if (allRoadLines[j].getPointsOnLine()[1].getId() == allRoadPoints[i].getId() && allRoadLines[j].getPointsOnLine()[0].getId() == allRoadPoints[i].getNeighbors()[0]) {
                foundSameLine = true;
                break;
            }
            if (allRoadLines[j].getPointsOnLine()[0].getId() == allRoadPoints[i].getId() && allRoadLines[j].getPointsOnLine()[1].getId() == allRoadPoints[i].getNeighbors()[0]) {
                foundSameLine = true;
                break;
            }
        }
        if (foundSameLine) {
            continue;
        }
        else {
            allRoadLines.push_back(MyLine(allRoadPoints[i], getMyPointWithId(allRoadPoints[i].getNeighbors()[0], allRoadPoints, deletedIDVec), lineId));
            lineId++;
        }
    }
    
    cout << "   - Total number of road points: " << allRoadPoints.size() << endl;
    cout << "   - Total number of road lines: " << allRoadLines.size() << endl;
    cout << "   - Finished in " << time(NULL) - currentTime << " seconds." << endl;
    
    
    cout << "3. Merging Nearby Road Points..." << endl;
    currentTime = time(NULL);
    
    // Traverse road point vector, find nearby points, use center point to replace the two old point
    float threshold_distance = P2P_THRESHOLD;
    int mergeCount = 0;
    if (allRoadPoints.size() > 0) {
        for (int i = 0; i < allRoadPoints.size() - 1; i++) {
            for (int j = i + 1; j < allRoadPoints.size(); j++) {
                MyPoint p1 = allRoadPoints[i];
                MyPoint p2 = allRoadPoints[j];
                if (point2PointDistance(p1, p2) < threshold_distance) {
                    int centerOfX = (p1.getPositionX() + p2.getPositionX()) / 2;
                    int centerOfY = (p1.getPositionY() + p2.getPositionY()) / 2;
                    allRoadPoints[i].setPosition(centerOfX, centerOfY);
                    allRoadPoints[j].setPosition(centerOfX, centerOfY);
                    mergeCount++;
                }
            }
        }
    }
    cout << "   - Merged for " << mergeCount << " times." << endl;
    cout << "   - Finished in " << time(NULL) - currentTime << " seconds." << endl;
    
    for (int i = 0; i < allRoadPoints.size(); i++) {
        if (allRoadPoints[i].getNeighbors().size() != 1) {
            cout << "Found one strange!" << endl;
        }
    }
    
    // remove overlapping points
    cout << "4. Removing Overlapping Points..." << endl;
    currentTime = time(NULL);
    vector<MyPoint> noDuplicateRoadPoints;
    
    noDuplicateRoadPoints.push_back(allRoadPoints.back());
    allRoadPoints.pop_back();
    while (!allRoadPoints.empty()) {
        bool foundDuplicate = false;
        for (int i = 0; i < noDuplicateRoadPoints.size(); i++) {
            if (allRoadPoints.back().getPositionX() == noDuplicateRoadPoints[i].getPositionX() && allRoadPoints.back().getPositionY() == noDuplicateRoadPoints[i].getPositionY()) {
                foundDuplicate = true;
                noDuplicateRoadPoints[i].addNeighbor(allRoadPoints.back().getNeighbors()[0]);   // share neighbor with each other
                ID4DeletedPoint obj;
                obj.deletedID = allRoadPoints.back().getId();
                obj.newID = noDuplicateRoadPoints[i].getId();
                deletedIDVec.push_back(obj);
                break;
            }
        }
        if (foundDuplicate) {
            allRoadPoints.pop_back();
        }
        else {
            noDuplicateRoadPoints.push_back(allRoadPoints.back());
            allRoadPoints.pop_back();
        }
    }
    cout << "   - " << noDuplicateRoadPoints.size() << " points remain." << endl;
    cout << "   - Finished in " << time(NULL) - currentTime << " seconds." << endl;
    
    
    // update point information in vectors of all points and lines
    cout << "5. Updating Point & Line information..." << endl;
    currentTime = time(NULL);
    
    // update points neighborhood
    for (int i = 0; i < noDuplicateRoadPoints.size(); i++) {
        for (int j = 0; j < noDuplicateRoadPoints[i].getNeighbors().size(); j++) {
            for (int k = 0; k < deletedIDVec.size(); k++) {
                if (noDuplicateRoadPoints[i].getNeighbors()[j] == deletedIDVec[k].deletedID) {
                    noDuplicateRoadPoints[i].removeNeighbor(deletedIDVec[k].deletedID);
                    noDuplicateRoadPoints[i].addNeighbor(deletedIDVec[k].newID);
                }
            }
        }
    }
    
    // update line info
    for (int i = 0; i < allRoadLines.size(); i++) {
        vector<MyPoint> twoPoints = allRoadLines[i].getPointsOnLine();
        int firstID = twoPoints[0].getId();
        int secondID = twoPoints[1].getId();
        bool idChanged = false;
        for (int j = 0; j < deletedIDVec.size(); j++) {
            if (firstID == deletedIDVec[j].deletedID) {
                firstID = deletedIDVec[j].newID;
                idChanged = true;
            }
            if (secondID == deletedIDVec[j].deletedID) {
                secondID = deletedIDVec[j].newID;
                idChanged = true;
            }
        }
        if (idChanged) {
            allRoadLines[i].setTwoPointsForLine(getMyPointWithId(firstID, noDuplicateRoadPoints, deletedIDVec), getMyPointWithId(secondID, noDuplicateRoadPoints, deletedIDVec));
        }
    }
    cout << "   - Finished in " << time(NULL) - currentTime << " seconds." << endl;
    
    // remove duplicate lines
    cout << "6. Removing Duplicate Lines..." << endl;
    currentTime = time(NULL);
    for (int i = 0; i < allRoadLines.size() - 1; i++) {
        for (int j = i + 1; j < allRoadLines.size(); j++) {
            MyLine l1 = allRoadLines[i];
            MyLine l2 = allRoadLines[j];
            int p1_id = l1.getPointsOnLine()[0].getId();
            int p2_id = l1.getPointsOnLine()[1].getId();
            int p3_id = l2.getPointsOnLine()[0].getId();
            int p4_id = l2.getPointsOnLine()[1].getId();
            if ((p1_id == p3_id && p2_id == p4_id) || (p1_id == p4_id && p2_id == p3_id)) {
                allRoadLines.erase(allRoadLines.begin() + j);
                if (j > i + 1) {
                    j--;
                }
                else {
                    j = i + 1;
                }
            }
        }
    }
    cout << "   - " << allRoadLines.size() << " lines remain." << endl;
    cout << "   - Finished in " << time(NULL) - currentTime << " seconds." << endl;
    
    // merge points with nearby lines
    cout << "7. Merging Points with Nearby Lines..." << endl;
    currentTime = time(NULL);
    // points vector is noDuplicateRoadPoints
    // lines vector is allRoadLines
    for (int i = 0; i < noDuplicateRoadPoints.size(); i++) {    // for each point
        for (int j = 0; j < allRoadLines.size(); j++) {         // for each line
            /************ be careful when line is vertical *************/
            MyPoint p = noDuplicateRoadPoints[i];
            MyLine l = allRoadLines[j];
            if (p.getId() == l.getPointsOnLine()[0].getId() || p.getId() == l.getPointsOnLine()[1].getId()) {
                continue;
            }
            vector<MyPoint> twoPoints = l.getPointsOnLine();
            Vec4i line = Vec4i(twoPoints[0].getPositionX(), twoPoints[0].getPositionY(), twoPoints[1].getPositionX(), twoPoints[1].getPositionY());
            float p2lDistance = point2LineDistance(Point(p.getPositionX(), p.getPositionY()), line);
            float lengthOfLine = point2PointDistance(twoPoints[0], twoPoints[1]);
            bool condition1 = (point2PointDistance(p, MyPoint(-1, (twoPoints[0].getPositionX() + twoPoints[1].getPositionX()) / 2, (twoPoints[0].getPositionY() + twoPoints[1].getPositionY()) / 2)) < (lengthOfLine / 2));     // distance from point to center -- Radius
            bool condition2 = (p2lDistance < P2L_THRESHOLD);     // distance from point to line
            if (condition1 && condition2) {
                // break line, use center point to form two new lines
                int centerPointX = -1;
                int centerPointY = -1;
                if (twoPoints[0].getPositionX() != twoPoints[1].getPositionX()) {
                    // if line is not vertical
                    float k = float(twoPoints[1].getPositionY() - twoPoints[0].getPositionY()) / float(twoPoints[1].getPositionX() - twoPoints[0].getPositionX());
                    if (twoPoints[1].getPositionY() == twoPoints[0].getPositionY()) {
                        // if origin line is horizon, new line will be vertical, formula: y = c
                        centerPointX = p.getPositionX();
                        centerPointY = (p.getPositionY() + twoPoints[0].getPositionY()) / 2;
                    }
                    else {
                        // both line are not horizon or vertical
                        k = -(1 / k);
                        float b = p.getPositionY() - k * p.getPositionX();
                        // now two lines are perpendicular to each other, one is MyLine l, the other is y = kx + b
                        // take the intersection point
                        Point intersectionPoint = getIntersectionOfTwoLines(k, b, l);
                        centerPointX = (p.getPositionX() + intersectionPoint.x) / 2;
                        centerPointY = (p.getPositionY() + intersectionPoint.y) / 2;
                    }
                }
                else {
                    // line is vertical
                    centerPointX = (p.getPositionX() + twoPoints[0].getPositionX()) / 2;
                    centerPointY = p.getPositionY();
                }
                
                if (twoPoints[0].getId() == -1 || twoPoints[1].getId() == -1) {
                    cout << "Wrong ID for two points of line!!" << endl;
                    exit(-1);
                }
                if (centerPointX < 0 || centerPointY < 0) {
                    cout << "Wrong new point position!" << endl;
                    exit(-1);
                }
                
                // update position for MyPoint p
                noDuplicateRoadPoints[i].setPosition(centerPointX, centerPointY);
                
                // update neighbors for this three points
                // a. move point to center point, and add neighbors
                int lineStartPointId = twoPoints[0].getId();
                int lineEndPointId = twoPoints[1].getId();
                if (lineStartPointId == -1 || lineEndPointId == -1) {
                    cout << "Wrong ID for two points of line!!" << endl;
                    exit(-1);
                }
                noDuplicateRoadPoints[i].addNeighbor(lineStartPointId);
                noDuplicateRoadPoints[i].addNeighbor(lineEndPointId);
                
                // b. break line by delete each other from neighbors
                noDuplicateRoadPoints[getIndexOfMyPointWithId(lineStartPointId, noDuplicateRoadPoints, deletedIDVec)].removeNeighbor(lineEndPointId);
                noDuplicateRoadPoints[getIndexOfMyPointWithId(lineEndPointId, noDuplicateRoadPoints, deletedIDVec)].removeNeighbor(lineStartPointId);
                
                // c. add new point to two old points' neighbor list
                noDuplicateRoadPoints[getIndexOfMyPointWithId(lineStartPointId, noDuplicateRoadPoints, deletedIDVec)].addNeighbor(noDuplicateRoadPoints[i].getId());
                noDuplicateRoadPoints[getIndexOfMyPointWithId(lineEndPointId, noDuplicateRoadPoints, deletedIDVec)].addNeighbor(noDuplicateRoadPoints[i].getId());
                
                // d. modify line in allRoadLines from AB to AO and then add line OB to the vector
                bool hasLineOne = false;
                bool hasLineTwo = false;
                for (int lNum = 0; lNum < allRoadLines.size(); lNum++) {
                    MyLine tempLine = allRoadLines[lNum];
                    int pt1 = tempLine.getPointsOnLine()[0].getId();
                    int pt2 = tempLine.getPointsOnLine()[1].getId();
                    if ((pt1 == lineStartPointId && pt2 == p.getId()) || (pt2 == lineStartPointId && pt1 == p.getId())) {
                        hasLineOne = true;
                    }
                    if ((pt1 == p.getId() && pt2 == lineEndPointId) || (pt2 == p.getId() && pt1 == lineEndPointId)) {
                        hasLineTwo = true;
                    }
                }
                if (!hasLineOne) {
                    allRoadLines[j].setTwoPointsForLine(getMyPointWithId(lineStartPointId, noDuplicateRoadPoints, deletedIDVec), noDuplicateRoadPoints[i]);
                }
                else {
                    allRoadLines.erase(allRoadLines.begin() + j);
                    j--;
                }
                if (!hasLineTwo) {
                    allRoadLines.push_back(MyLine(noDuplicateRoadPoints[i], getMyPointWithId(lineEndPointId, noDuplicateRoadPoints, deletedIDVec), -10));    // assign id = -10 to new connected lines
                }
                
                j = -1;
            }
        }
    }
    cout << "   - Finished in " << time(NULL) - currentTime << " seconds." << endl;
    
    /////////////////////////////////////////////////////
    
    cout << "8. Merging Remaining Nearby Road Points..." << endl;
    currentTime = time(NULL);
    
    // Traverse road point vector, find nearby points, use center point to replace the two old point
    threshold_distance = P2P_THRESHOLD;
    mergeCount = 0;
    for (int i = 0; i < noDuplicateRoadPoints.size() - 1; i++) {
        for (int j = i + 1; j < noDuplicateRoadPoints.size(); j++) {
            MyPoint p1 = noDuplicateRoadPoints[i];
            MyPoint p2 = noDuplicateRoadPoints[j];
            if (point2PointDistance(p1, p2) < threshold_distance) {
                int centerOfX = (p1.getPositionX() + p2.getPositionX()) / 2;
                int centerOfY = (p1.getPositionY() + p2.getPositionY()) / 2;
                noDuplicateRoadPoints[i].setPosition(centerOfX, centerOfY);
                noDuplicateRoadPoints[j].setPosition(centerOfX, centerOfY);
                mergeCount++;
            }
        }
    }
    cout << "   - Merged for " << mergeCount << " times." << endl;
    cout << "   - Finished in " << time(NULL) - currentTime << " seconds." << endl;
    
    
    /////////////////////////////////////////////////////
    
    // draw lines on image
    cout << "9. Saving Result..." << endl;
    Mat outImg = Mat::zeros(IMAGE_TILE_SIZE.height, IMAGE_TILE_SIZE.width, 0);
    vector<vector<Point>> pointPairsToDraw;
    
    for (int i = 0; i < allRoadLines.size(); i++) {
        // draw with allRoadLines
        vector<MyPoint> twoPoints = allRoadLines[i].getPointsOnLine();
        MyPoint p1 = getMyPointWithId(twoPoints[0].getId(), noDuplicateRoadPoints, deletedIDVec);
        MyPoint p2 = getMyPointWithId(twoPoints[1].getId(), noDuplicateRoadPoints, deletedIDVec);
        if (p1.getNeighbors().size() == 1 && p2.getNeighbors().size() == 1) {
            continue;
        }
        int p1_x = p1.getPositionX();
        int p1_y = p1.getPositionY();
        int p2_x = p2.getPositionX();
        int p2_y = p2.getPositionY();
        line(outImg, Point(p1_x, p1_y), Point(p2_x, p2_y), Scalar(255), LINE_AA);
    }
    if (OUTPUT_PATH.empty()) {
        createFolder("../results/" + MODEL_NAME + "/post_processing_result/");
        imwrite("../results/" + MODEL_NAME + "/post_processing_result/" + cityName + ".png", outImg);
    } else {
        createFolder(OUTPUT_PATH);
        imwrite(OUTPUT_PATH + cityName + ".png", outImg);
    }
    
    cout << "DONE!\nTime used: " << time(NULL) - startTime << " seconds." << endl;
}

float point2PointDistance(MyPoint p1, MyPoint p2) {
    float distance = pow(p1.getPositionX() - p2.getPositionX(), 2) + pow(p1.getPositionY() - p2.getPositionY(), 2);
    return sqrt(distance);
}

vector<vector<Vec4i>> houghLineOnPatch(string cityName, string MODEL_NAME, string INPUT_PATH, Size IMAGE_TILE_SIZE, int MAX_PATCH_LOC_X, int MAX_PATCH_LOC_Y) {
    string rootPath = BASE_PATH;
    string directory = rootPath + MODEL_NAME + "/result_on_patches/";   // model_name needs to be changed
    vector<vector<Vec4i>> allLines;
    for (int patch_position_x = 0; patch_position_x < MAX_PATCH_LOC_X; patch_position_x++) {
        for (int patch_position_y = 0; patch_position_y < MAX_PATCH_LOC_Y; patch_position_y++) {
            string fileName = to_string(patch_position_x) + "_" + to_string(patch_position_y) + ".png";
            //Mat rgbImg = imread(directory + cityName + "_" + fileName);
            Mat predImg = imread(directory + cityName + "_" + fileName, IMREAD_GRAYSCALE);
            if (!INPUT_PATH.empty()) {
                Mat predImg = imread(INPUT_PATH + cityName + "_" + fileName, IMREAD_GRAYSCALE);
            }
            if (!predImg.data) {
                cout << "Failed to load image patches, wrong input path!" << endl;
                exit(-1);
            }
            Mat dst;    //, dstHL = Mat::zeros(Size(200, 200), CV_32FC1);
            
            Canny(predImg, dst, 50, 200, 3);
            
            vector<Vec4i> lines;
            HoughLinesP(dst, lines, 1, CV_PI / 180, 50, 100, 50);

            if (lines.size() > 1) {
                for (int i = 0; i < lines.size() - 1; i++) {
                    for (int j = i + 1; j < lines.size(); j++) {
                        if (isParallelLine(lines[i], lines[j])) {
                            float len_i = pow(lines[i][0] - lines[i][2], 2) + pow(lines[i][1] - lines[i][3], 2);
                            float len_j = pow(lines[j][0] - lines[j][2], 2) + pow(lines[j][1] - lines[j][3], 2);
                            if (len_j < len_i) {
                                lines.erase(lines.begin() + j);
                            }
                            else {
                                lines.erase(lines.begin() + i);
                            }
                            i = 0;
                            j = 0;
                        }
                    }
                }
            }
            
            // cordination transfer from patch(200 x 200) to image(8192 x 8192)
            for (int i = 0; i < lines.size(); i++) {
                if (patch_position_x == MAX_PATCH_LOC_X - 1) {
                    lines[i][0] = lines[i][0] + (IMAGE_TILE_SIZE.height - 1) - 200;   // (8192-1) -200
                    lines[i][2] = lines[i][2] + (IMAGE_TILE_SIZE.height - 1) - 200;
                }
                else {
                    lines[i][0] = lines[i][0] + patch_position_x * 100;    // step size 100 or 50
                    lines[i][2] = lines[i][2] + patch_position_x * 100;
                }
                if (patch_position_y == MAX_PATCH_LOC_Y - 1) {
                    lines[i][1] = lines[i][1] + (IMAGE_TILE_SIZE.width - 1) - 200;   // (8192-1) -200
                    lines[i][3] = lines[i][3] + (IMAGE_TILE_SIZE.width - 1) - 200;
                }
                else {
                    lines[i][1] = lines[i][1] + patch_position_y * 100;
                    lines[i][3] = lines[i][3] + patch_position_y * 100;
                }
            }
            
            allLines.push_back(lines);
        }
    }
    return allLines;
}

bool isParallelLine(Vec4i l1, Vec4i l2) {
    bool isParallel;
    if (abs(l1[0] - l1[2]) > 15 && abs(l2[0] - l2[2]) > 15) {
        float d1 = point2LineDistance(Point(l1[0], l1[1]), l2);
        float d2 = point2LineDistance(Point(l1[2], l1[3]), l2);
        if (d1 < 50 && d2 < 50) {
            isParallel = true;
        }
        else {
            isParallel = false;
        }
    }
    else {    // vertical lines
        if (abs(l1[0] - l2[0]) > 50 && abs(l1[2] - l2[2]) > 50) {
            isParallel = false;
        }
        else {
            isParallel = true;
        }
    }
    return isParallel;
}

float point2LineDistance(Point p, Vec4i line) {
    float distance;
    float x1 = line[0];
    float y1 = line[1];
    float x2 = line[2];
    float y2 = line[3];
    
    if (x1 == x2) {
        // line is vertical
        return abs(y1 - y2);
    }
    float A = (y2 - y1) / (x2 - x1);
    float B = -1.0;
    float C = y1 - x1 * A;
    
    distance = abs(A * p.x + B * p.y + C) / sqrt(pow(A, 2) + pow(B, 2));
    
    return distance;
}

MyPoint getMyPointWithId(int pointId, vector<MyPoint> allPointsVec, vector<ID4DeletedPoint> deletedIDVec) {
    int index = -1;
    for (int i = 0; i < deletedIDVec.size(); i++) {
        if (pointId == deletedIDVec[i].deletedID) {
            pointId = deletedIDVec[i].newID;
            break;
        }
    }
    for (int i = 0; i < allPointsVec.size(); i++) {
        if (allPointsVec[i].getId() == pointId) {
            index = i;
        }
    }
    if (index == -1) {
        cout << "Point with ID " << pointId << " Not Found!" << endl;
        exit(-1);
    }
    return allPointsVec[index];
}

int getIndexOfMyPointWithId(int pointId, vector<MyPoint> allPointsVec, vector<ID4DeletedPoint> deletedIDVec) {
    int index = -1;
    for (int i = 0; i < deletedIDVec.size(); i++) {
        if (pointId == deletedIDVec[i].deletedID) {
            pointId = deletedIDVec[i].newID;
            break;
        }
    }
    for (int i = 0; i < allPointsVec.size(); i++) {
        if (allPointsVec[i].getId() == pointId) {
            index = i;
        }
    }
    if (index == -1) {
        cout << "Point with ID " << pointId << " Not Found!" << endl;
        exit(-1);
    }
    return index;
}

Point getIntersectionOfTwoLines(float k, float b, MyLine l) {
    // line 1 is y = kx +b
    // line 2 is l, which contains two known points
    float x1 = l.getPointsOnLine()[0].getPositionX();
    float y1 = l.getPointsOnLine()[0].getPositionY();
    float x2 = l.getPointsOnLine()[1].getPositionX();
    float y2 = l.getPointsOnLine()[1].getPositionY();
    
    float A = float(y2 - y1) / float(x2 - x1);
    //    float B = -1.0;
    float C = y1 - x1 * A;
    
    float temp = (b - C) / (A - k);
    int x = round((b - C) / (A - k));
    int y = round(k * temp + b);
    return Point(x, y);
}

void drawDiffMapOnRGB(vector<string> cities){
    string rootPath = BASE_PATH;
    string directory = "my_model/post_processing_result/";  // change model_name
    for (int i = 0; i < cities.size(); i++) {
        Mat diffImg = imread(rootPath + directory + "errorImg/" + cities[i] + ".png");
        Mat rgbImg = imread("../data/rgb_ng/" + cities[i] + ".png");
        if (!rgbImg.data) {
            cout << "Failed to load RGB imagery, wrong input path!" << endl;
            exit(-1);
        }
        for (int i = 0; i < diffImg.rows; i++) {
            for (int j = 0; j < diffImg.cols; j++) {
                if (diffImg.at<Vec3b>(i, j) != Vec3b(0, 0, 0)) {
                    rgbImg.at<Vec3b>(i, j) = diffImg.at<Vec3b>(i, j);
                }
            }
        }
        
        imwrite(rootPath + directory + "errorImg/" + cities[i] + "_rgb.png", rgbImg);
    }
}

void generateAllPatches(vector<string> cities, string INPUT_PATH, string OUTPUT_PATH) {
    for (int idx = 0; idx < cities.size(); idx++) {
        string path_in = "rgb_ng/";
        string path_out = "rgb_ng/patches_to_predict/";
        Mat img = imread("../data/" + path_in + cities[idx] + ".png");
        if (!INPUT_PATH.empty()) {
            Mat img = imread(INPUT_PATH + cities[idx] + ".png");
        }
        if (!img.data) {
            cout << "Failed to load RGB imagery, wrong input path!" << endl;
            exit(-1);
        }
        int patchSize = 200;
        int i_count = 0, j_count = 0;
        int overlapSize = 100;
        int increaseStep = patchSize - overlapSize;
        bool lastTripI = false;
        bool lastTripJ = false;
        for (int i = 0; i < img.rows; i += increaseStep) {
            if (i + patchSize > img.rows) {
                i = img.rows - patchSize;
                lastTripI = true;
            }
            for (int j = 0; j < img.cols; j += increaseStep) {
                if (j + patchSize > img.cols) {
                    j = img.cols - patchSize;
                    lastTripJ = true;
                }
                Mat patchImg(img, Rect(i, j, patchSize, patchSize));
                if (OUTPUT_PATH.empty()) {
                    createFolder("../data/" + path_out);
                    imwrite("../data/" + path_out + cities[idx] + "_" + to_string(i_count) + "_" + to_string(j_count) + ".png", patchImg);
                } else {
                    createFolder(OUTPUT_PATH);
                    imwrite(OUTPUT_PATH + cities[idx] + "_" + to_string(i_count) + "_" + to_string(j_count) + ".png", patchImg);
                }
                j_count++;
                if (lastTripJ) {
                    break;
                }
            }
            i_count++;
            j_count = 0;
            lastTripJ = false;
            if (lastTripI) {
                break;
            }
        }
    }
}

void startEval(vector<string> cities) {
    cout << "   - Generating difference map..." << endl;
    generateErrorImage(cities);
    cout << "   - Evaluating..." << endl;
    evaluateError(cities);
}

void generateErrorImage(vector<string> cities) {
    // Load truthImg, predImg
    for (int idx = 0; idx < cities.size(); idx++) {
        cout << "       - " << cities[idx] << endl;
        string rootFolder = BASE_PATH;
        string baseFolder = rootFolder + "my_model/post_processing_result/";    // change model_name
        Mat truthImg = imread("../data/y_ng/" + cities[idx] + ".png", IMREAD_GRAYSCALE);
        Mat predImg = imread(baseFolder + cities[idx] + ".png", IMREAD_GRAYSCALE);
        if (!truthImg.data || !predImg.data) {
            cout << "Failed to load images, wrong input path!" << endl;
            exit(-1);
        }
        Mat rc = Mat::zeros(truthImg.rows, truthImg.cols, 0);   // red channel
        Mat img;
        vector<Mat> imgCs;
        imgCs.push_back(truthImg);  // B
        imgCs.push_back(predImg);   // G
        imgCs.push_back(rc);        // R
        merge(imgCs, img);
        for (int j = 0; j < img.rows; j++) {
            for (int k = 0; k < img.cols; k++) {
                if (img.at<Vec3b>(j, k)[0] == 0 && img.at<Vec3b>(j, k)[1] != 0) {   // BGR:(0, 1, 0) :: GREEN, predict
                    img.at<Vec3b>(j, k) = Vec3b(0, 0, 255);
                } else if (img.at<Vec3b>(j, k)[0] != 0 && img.at<Vec3b>(j, k)[1] == 0) {    // BGR:(1, 0, 0) :: GREEN, truth
                    ;
                } else if (img.at<Vec3b>(j, k)[0] != 0 && img.at<Vec3b>(j, k)[1] != 0) {    // BGR:(1, 1, 0) :: truth-positive, --> green
                    img.at<Vec3b>(j, k) = Vec3b(0, 255, 0);
                }
            }
        }
        
        vector<Mat> rgbVec;
        split(img, rgbVec);
        Mat falsePositiveImg = rgbVec[2];          // r
        Mat compareImg = rgbVec[0] + rgbVec[1];    // g + b

        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                if (falsePositiveImg.at<uchar>(i, j) > 0 && searchAround(i, j, compareImg)) {
                    img.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
                }
                if (rgbVec[0].at<uchar>(i, j) > 0 && searchAround(i, j, rgbVec[1])) {
                    img.at<Vec3b>(i, j) = Vec3b(0, 0, 0);
                }
            }
        }
        imwrite(baseFolder + "errorImg/" + cities[idx] + ".png", img);

    }
    
}

void evaluateError(vector<string> cities) {
    
    string rootFolder = BASE_PATH;
    string baseFolder = rootFolder + "my_model/post_processing_result/errorImg/";   // change model_name
    ofstream out_file(baseFolder + "eval.txt");
    
    out_file << "City Names\tPrecision\tRecall\tF1\tIoU\n";
    
    for (int idx = 0; idx < cities.size(); idx++) {
        
        cout << "       - " << cities[idx] << endl;
        
        Mat img = imread(baseFolder + cities[idx] + ".png");
        if (!img.data) {
            cout << "Failed to load RGB imagery, wrong input path!" << endl;
            exit(-1);
        }
        vector<Mat> rgbImg;
        split(img, rgbImg);
        
        // TP: green
        // FP: red
        // FN: blue
        int FP = countNonZero(rgbImg[2]);
        int TP = countNonZero(rgbImg[1]);
        int FN = countNonZero(rgbImg[0]);
        
        // precision = TP / (TP + FP)
        // recall =
        // F1 = P * R * 2 / (P + R)
        // IoU = TP / (TP + FP + FN)
        float precision = float(TP) / float(TP + FP);
        float recall = float(TP) / float(TP + FN);
        float F1 = precision * recall * 2 / (precision + recall);
        float iou = float(TP) / float(TP + FP + FN);
        
        out_file << cities[idx] << "\t" << precision << "\t" << recall << "\t" << F1 <<  "\t" << iou << "\n";
        
    }
    
    out_file.close();
    
}

bool searchAround(int rowIdx, int colIdx, Mat templateImg) {

    int startRowIdx, startColIdx;
    int searchRange = 20;
    if (rowIdx < searchRange) {
        startRowIdx = 0;
    } else if (rowIdx > templateImg.rows-searchRange) {
        startRowIdx = templateImg.rows-searchRange*2;
    } else {
        startRowIdx = rowIdx-searchRange;
    }
    if (colIdx < searchRange) {
        startColIdx = 0;
    } else if (colIdx > templateImg.cols-searchRange) {
        startColIdx = templateImg.cols-searchRange*2;
    } else {
        startColIdx = colIdx-searchRange;
    }
    
    for (int i = startRowIdx; i < startRowIdx + searchRange*2-2; i++) {
        for (int j = startColIdx; j < startColIdx + searchRange*2-2; j++) {
            if (templateImg.at<uchar>(i, j) > 0) {
                return true;
            }
        }
    }
    
    return false;
}


