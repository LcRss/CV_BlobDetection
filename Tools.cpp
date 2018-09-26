//
// Created by luca on 26/06/18.
//

#include "Tools.h"

using namespace std;
using namespace cv;

Tools::Tools() = default;

/**
 * Loads from all the image from that path
 *
 * @param path_img
 * @return
 */

vector<Mat> Tools::load(const string &path_img) {

    vector<String> filename;
    vector<Mat> setImages;

    glob(path_img, filename, true);
    for (size_t k = 0; k < filename.size(); ++k) {

        Mat im = imread(filename[k], CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
        if (im.empty()) continue;
        setImages.push_back(im);

    }

    return setImages;
}

/**
 * Create a Foreground Mask for each image,
 *  making the absdiff between the empty image and the populated image
 *
 * @param imgs vector with all depth maps images
 * @return
 */

vector<Mat> Tools::findMaskFromDepth(vector<Mat> &imgs) {

    vector<Mat> results;

    Mat background = imgs[0];

    for (int i = 1; i < imgs.size(); ++i) {

        Mat people = imgs[i];

        Mat diffImage;
        absdiff(background, people, diffImage);

        double minVal;
        double maxVal;
        minMaxLoc(diffImage, &minVal, &maxVal);

        Mat mask;
        inRange(diffImage, 1000, maxVal, mask);

        results.push_back(mask);
    }

    return results;
}

/**
 * Performs the morphological transformations opening to remove some noise from the image
 *  erosion + dilation
 *
 * @param imgs vector with all the created mask in the previous step
 * @param ksize size of the structuring element.
 * @return
 */

vector<Mat> Tools::refineMask(vector<Mat> &imgs, int ksize) {

    vector<Mat> results;

    int morphElem = 2;
    int morphSize = ksize;
    int morphOperator = 0;
    int operation = morphOperator + 2;

    for (const auto &img : imgs) {

        Mat element = getStructuringElement(morphElem, Size(2 * morphSize + 1, 2 * morphSize + 1),
                                            Point(morphSize, morphSize));
        Mat dst;
        morphologyEx(img, dst, operation, element);
        results.push_back(dst);
    }

    return results;
}

/**
 * Applies Masks on the depth map images.
 *
 * @param imgs vector of depth map images
 * @param mask vector of masks
 * @return
 */

vector<Mat> Tools::applyMask(vector<Mat> &imgs, vector<Mat> &mask) {

    vector<Mat> results;

    for (int i = 1; i < imgs.size(); ++i) {

        Mat tmp;
        Mat eq;
        equalizeHist(imgs[i], eq);
        eq.copyTo(tmp, mask[i - 1]);
        results.push_back(tmp);
    }

    return results;

}

/**
 * Normalize every depth map in order to corretly visualize it
 *
 * @param imgs vector of depth maps
 * @return
 */

vector<Mat> Tools::normalizeDepthImg(vector<Mat> &imgs) {

    vector<Mat> results;

    for (int i = 0; i < imgs.size(); ++i) {

        Mat norm;
        normalize(imgs[i], norm, 0, 255, CV_MINMAX);
        norm.convertTo(norm, CV_8UC1);
        results.push_back(norm);
    }

    return results;
}

/**
 * Set to white the background pixels of each image
 *
 * @param imgs vector of images
 * @return
 */

vector<Mat> Tools::whiteBackground(std::vector<cv::Mat> &imgs) {

    vector<Mat> results;

    for (auto &img : imgs) {

        Mat tmp;

        Mat res = img.clone();

        inRange(res, 0, 0, tmp);
        res.setTo(255, tmp);

        results.push_back(res);

    }

    return results;
}

vector<Mat> Tools::otherPr(std::vector<cv::Mat> &imgs) {

    vector<Mat> results;

    for (auto &img : imgs) {

        Mat res;

        copyMakeBorder(img, res, 20, 20, 20, 20, BORDER_CONSTANT, 255);
//        imshow("prova",res);
//        waitKey(0);
        results.push_back(res);

    }

    return results;


}

/**
 * Uses SimpleBolbDetector to find Blob
 *  filer by Color
 *
 * @param imgs vector of images from the previus step whiteBackground.
 * @return the keypoints of each blob
 */

vector<vector<KeyPoint>> Tools::findBlob(std::vector<cv::Mat> &imgs) {

    vector<vector<KeyPoint>> finalKPs;

    SimpleBlobDetector::Params params;


    params.minRepeatability = 2;

    params.filterByColor = true;
    params.filterByArea = false;
    params.filterByCircularity = false;
    params.filterByConvexity = false;
    params.filterByInertia = false;

    std::vector<int> values = {1, 15};

    for (int i = 0; i < imgs.size(); ++i) {

        double minVal;
        double maxVal;
        minMaxLoc(imgs[i], &minVal, &maxVal);

        params.minThreshold = 0;
        params.maxThreshold = (float) minVal + 30;

        vector<KeyPoint> refineKP;

        for (auto &v : values) {

            params.thresholdStep = v;
            Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

            vector<KeyPoint> keypoints;
            detector->detect(imgs[i], keypoints);

            for (auto &keypoint : keypoints) {
                if (keypoint.size > 30 && keypoint.size < 610)
                    refineKP.push_back(keypoint);
            }

        }

        vector<KeyPoint> kpNotClose;

        for (const auto &j : refineKP) {

            if (!kpNotClose.empty()) {
                bool ins = true;

                for (auto &keypoint : kpNotClose) {

                    float r = keypoint.overlap(keypoint, j);

                    if (r != 0) {
                        ins = false;
                        break;
                    }

                }

                if (ins) {
                    kpNotClose.push_back(j);
                }

            } else {
                kpNotClose.push_back(j);
            }

        }

        refineKP = kpNotClose;

        Mat im_with_keypoints;
        drawKeypoints(imgs[i], refineKP, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cout << "Image # " + to_string(i) + "; People # " + to_string(refineKP.size()) << endl;

        finalKPs.push_back(refineKP);


        string n = "Keypoints - People Image # " + to_string(i) + ".png";

        cv::Mat image = im_with_keypoints;
        cv::Rect border(cv::Point(0, 0), image.size());
        cv::Scalar color(0, 0, 0);
        int thickness = 3;

        cv::rectangle(image, border, color, thickness);
        imwrite(n, image);


        imshow(n, im_with_keypoints);
        waitKey(0);

    }

    return finalKPs;

}

/**
 * It is used to print all the images in a vector
 *  In case of depth maps as input process them in order to correctly visualize
 *
 * @param imgs
 * @param depth
 */

void Tools::print(vector<Mat> &imgs, bool depth, string &nameInput) {

    if (depth) {

        for (int k = 0; k < imgs.size(); ++k) {

            Mat depthVis;
            normalize(imgs[k], depthVis, 0, 255, CV_MINMAX);
            depthVis.convertTo(depthVis, CV_8UC1);

            string name = "Depth map Image - " + to_string(k) + ".png";
//            imwrite(name, depthVis);

            namedWindow(name);
            imshow(name, depthVis);

            waitKey(0);

        }

    } else {

        for (int k = 0; k < imgs.size(); ++k) {

            string name = nameInput + " image-" + to_string(k);

            namedWindow(name);
            imshow(name, imgs[k]);

            waitKey(0);

        }
    }

}


//These Methods don't create good mask

/**
 * Try to create masks throght BackgroundSubtractor
 *
 * @param imgs
 * @return
 */

vector<Mat> Tools::findMaskKNN(vector<Mat> &imgs) {

    vector<cv::Mat> results;

    Mat res;
    Ptr<BackgroundSubtractor> pKNN;
    pKNN = createBackgroundSubtractorKNN(500, 200.0, true);

    int i = 0;
    while (i < 500) {
        ++i;
        pKNN->apply(imgs[0], res);
    }

    for (int j = 1; j < imgs.size(); ++j) {

        pKNN->apply(imgs[0], res);
        pKNN->apply(imgs[j], res);

        threshold(res, res, 254, 255, THRESH_BINARY);
        results.push_back(res);

//        string name = "../KNN_" + to_string(j) + ".png";
//        imwrite(name, res);
//        imshow(to_string(j), res);
//        waitKey(0);

    }

    return results;
}

/**
 * Try to create masks throgth difference of color images
 *
 * @param imgs
 * @return
 */

std::vector<cv::Mat> Tools::findMaskFromBGR(std::vector<cv::Mat> &imgs) {

    vector<Mat> results;

    Mat diffImage;

    Mat background = imgs[0];

    for (int k = 1; k < imgs.size(); ++k) {


        absdiff(background, imgs[k], diffImage);
        Mat foregroundMask = Mat::zeros(diffImage.rows, diffImage.cols, CV_8UC1);

        float threshold = 90.0f;
        float dist;

        for (int j = 0; j < diffImage.rows; ++j) {
            for (int i = 0; i < diffImage.cols; ++i) {
                Vec3b pix = diffImage.at<Vec3b>(j, i);

                dist = (pix[0] * pix[0] + pix[1] * pix[1] + pix[2] * pix[2]);
                dist = sqrt(dist);

                if (dist > threshold) {
                    foregroundMask.at<unsigned char>(j, i) = 255;
                }
            }
        }

        results.push_back(foregroundMask);
    }
    return results;
}