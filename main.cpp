//
// Created by luca on 26/06/18.
//

#include "Tools.h"

using namespace cv;
using namespace std;


int main(int argc, char **argv) {

    Tools tools = Tools();
    //Pass to tools.print
    string m;

    String path_col_img(argv[1]);
    String path_dep_img(argv[2]);

    cout << "------------------------------------" << endl;
    cout << "Path:" << path_col_img << endl;
    cout << "Path:" << path_dep_img << endl;
    cout << "------------------------------------" << endl;

    cout << "LOADS IMAGES" << endl;
    cout << "------------------------------------" << endl;

    vector<Mat> col_img;
    col_img = tools.load(path_col_img);

    vector<Mat> dep_img;
    dep_img = tools.load(path_dep_img);

    cout << "FIND MASK FROM DEPTH MAPS" << endl;
    cout << "------------------------------------" << endl;

    vector<Mat> depthMask = tools.findMaskFromDepth(dep_img);
//    m = "Created Mask ";
//    tools.print(depthMask, false, m);
//    destroyAllWindows();

    cout << "REFINE MASK" << endl;
    cout << "------------------------------------" << endl;

    vector<Mat> refineMask = tools.refineMask(depthMask, 8);
//    m = "Refined Mask";
//    tools.print(refineMask, false, m);
//    destroyAllWindows();

    cout << "NORMALIZE THE RESULTS depth+mask" << endl;
    cout << "------------------------------------" << endl;

    vector<Mat> a = tools.normalizeDepthImg(dep_img);
//    m = "Normalize ";
//    tools.print(normDepth, false, m);
//    destroyAllWindows();

    cout << "APPLY MASK TO IMAGES" << endl;
    cout << "------------------------------------" << endl;

    vector<Mat> maskApplied = tools.applyMask(a, depthMask);
//    m = "Mask Applied to ";
//    tools.print(maskApplied, true, m);
//    destroyAllWindows();

//    string name = "../maskApp.png";
//    imwrite(name, maskApplied[0]);

    cout << "REFINE AGAIN" << endl;
    cout << "------------------------------------" << endl;

    vector<Mat> refined = tools.refineMask(maskApplied, 4);
//     m = "Refine Again ";
//    tools.print(refined, false, m);
//    destroyAllWindows();

//    name = "../maskApp.png";
//    imwrite(name, refined[0]);

    cout << "SETTING TO WHITE THE BACKGROUND" << endl;
    cout << "------------------------------------" << endl;

    vector<Mat> whiteBackGr = tools.whiteBackground(refined);
//    m = "WhiteBackground";
//    tools.print(whiteBackGr, false, m);
//    destroyAllWindows();

//    name = "../whiteBack.png";
//
//    cv::Mat image = whiteBackGr[0];
//    cv::Rect border(cv::Point(0, 0), image.size());
//    cv::Scalar color(0, 0, 0);
//    int thickness = 3;
//
//    cv::rectangle(image, border, color, thickness);
//    imwrite(name, image);

    vector<Mat> p = tools.otherPr(whiteBackGr);

    cout << "FIND BLOBS" << endl;
    cout << "------------------------------------" << endl;

    vector<vector<KeyPoint>> kp = tools.findBlob(p);
    destroyAllWindows();

    waitKey(0);
    return 0;
}