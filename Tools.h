//
// Created by luca on 26/06/18.
//

#ifndef PROJECTS_1_TOOLS_H
#define PROJECTS_1_TOOLS_H


#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv/cxmisc.h>
#include <opencv/cv.hpp>

#include <iostream>
#include <vector>
#include <glob.h>


class Tools {

public:

    explicit Tools();

    std::vector<cv::Mat> load(const std::string &path_img);

    std::vector<cv::Mat> findMaskFromDepth(std::vector<cv::Mat> &imgs);

    std::vector<cv::Mat> refineMask(std::vector<cv::Mat> &imgs, int ksize);

    std::vector<cv::Mat> applyMask(std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &mask);

    std::vector<cv::Mat> normalizeDepthImg(std::vector<cv::Mat> &imgs);

    std::vector<cv::Mat> whiteBackground(std::vector<cv::Mat> &imgs);

    std::vector<cv::Mat> otherPr(std::vector<cv::Mat> &imgs);

    std::vector<std::vector<cv::KeyPoint>> findBlob(std::vector<cv::Mat> &imgs);

    void print(std::vector<cv::Mat> &imgs, bool depth, std::string& nameInput);

    std::vector<cv::Mat> findMaskKNN(std::vector<cv::Mat> &imgs);

    std::vector<cv::Mat> findMaskFromBGR(std::vector<cv::Mat> &imgs);


};


#endif //PROJECTS_1_TOOLS_H
