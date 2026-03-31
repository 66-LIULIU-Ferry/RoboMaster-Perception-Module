#include "yolov7_kpt.h"

// 用车上的相机
#define CAMERA_ID 0

yolo_kpt DEMO;
std::vector<yolo_kpt::Object> result;
cv::TickMeter meter;

// 相机内参
cv::Mat K = (cv::Mat_<double>(3,3) <<
    800, 0, 320,
    0, 800, 240,
    0, 0, 1);

cv::Mat dist = cv::Mat::zeros(4,1,CV_64F);

// 装甲板实际尺寸（单位 mm）
std::vector<cv::Point3f> obj_pts = {
    {-65, -27, 0},
    { 65, -27, 0},
    { 65,  27, 0},
    {-65,  27, 0}
};

// 简单卡尔曼：状态[x,y,z,vx,vy,vz]，观测[x,y,z]
cv::KalmanFilter kf(6, 3);

void initKalman() {
    kf.transitionMatrix = (cv::Mat_<float>(6,6) <<
        1,0,0,1,0,0,
        0,1,0,0,1,0,
        0,0,1,0,0,1,
        0,0,0,1,0,0,
        0,0,0,0,1,0,
        0,0,0,0,0,1);

    setIdentity(kf.measurementMatrix);
    setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-4));
    setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-2));
    setIdentity(kf.errorCovPost, cv::Scalar::all(1));
}

int main() {

    initKalman();

    cv::VideoCapture cap;
    cap.open(CAMERA_ID);

    if (!cap.isOpened()) {
        std::cout << "摄像头未打开" << std::endl;
        return 0;
    }

    while (true) {
        cv::Mat src_img;
        if (!cap.read(src_img)) break;

        meter.start();
        result = DEMO.work(src_img);   // 这里返回 bbox + 构造的4个点
        meter.stop();

        printf("Time: %.2f ms\n", meter.getTimeMilli());
        meter.reset();

        for (auto& obj : result) {

            // 从检测结果里拿4个角点
            std::vector<cv::Point2f> img_pts = obj.kpt;
            if (img_pts.size() != 4) continue;

            // --- PnP ---
            cv::Mat rvec, tvec;
            bool ok = cv::solvePnP(obj_pts, img_pts, K, dist, rvec, tvec);
            if (!ok) continue;

            // --- Kalman ---
            cv::Mat measurement = (cv::Mat_<float>(3,1) <<
                (float)tvec.at<double>(0),
                (float)tvec.at<double>(1),
                (float)tvec.at<double>(2));

            cv::Mat predict = kf.predict();
            cv::Mat state   = kf.correct(measurement);

            float x = state.at<float>(0);
            float y = state.at<float>(1);
            float z = state.at<float>(2);

            // 画框
            cv::rectangle(src_img, obj.rect, cv::Scalar(0,255,0), 2);

            // 画四个点
            for (auto& p : img_pts)
                cv::circle(src_img, p, 3, cv::Scalar(0,0,255), -1);

            // 显示三维位置
            char buf[100];
            sprintf(buf, "X:%.1f Y:%.1f Z:%.1f", x, y, z);
            cv::putText(src_img, buf, cv::Point(20,50),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0,255,0), 2);
        }

        cv::imshow("demo", src_img);

        if (cv::waitKey(1) == 27) break;
    }

    return 0;
}