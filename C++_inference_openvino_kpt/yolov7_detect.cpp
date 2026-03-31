#include "yolov7_kpt.h"

// 构造函数：加载模型 
yolo_kpt::yolo_kpt() {
    model = core.read_model(MODEL_PATH);

    // 编译模型（OpenVINO）
    compiled_model = core.compile_model(model, DEVICE);

    // 创建推理请求
    infer_request = compiled_model.create_infer_request();

    // 获取输入tensor
    input_tensor1 = infer_request.get_input_tensor(0);
}


//  图像预处理（letterbox） 
cv::Mat yolo_kpt::letter_box(cv::Mat &src, int h, int w, std::vector<float> &padd) {
    int in_w = src.cols;
    int in_h = src.rows;

    float r = std::min(float(h) / in_h, float(w) / in_w);

    int new_w = round(in_w * r);
    int new_h = round(in_h * r);

    int pad_w = w - new_w;
    int pad_h = h - new_h;

    cv::Mat resized;
    resize(src, resized, cv::Size(new_w, new_h));

    pad_w /= 2;
    pad_h /= 2;

    padd = { (float)pad_w, (float)pad_h, r };

    cv::copyMakeBorder(resized, resized,
                       pad_h, pad_h,
                       pad_w, pad_w,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(114,114,114));

    return resized;
}


//  坐标还原 
cv::Rect yolo_kpt::scale_box(cv::Rect box, std::vector<float> &padd, float raw_w, float raw_h) {
    cv::Rect res;
    res.width  = box.width  / padd[2];
    res.height = box.height / padd[2];
    res.x = std::max(std::min((box.x - padd[0]) / padd[2], raw_w - 1.f), 0.f);
    res.y = std::max(std::min((box.y - padd[1]) / padd[2], raw_h - 1.f), 0.f);
    return res;
}


void yolo_kpt::drawPred(int classId, float conf, cv::Rect box,
                        std::vector<cv::Point2f> point,
                        cv::Mat &frame,
                        const std::vector<std::string> &classes) {

    // 画bbox
    cv::rectangle(frame, box, cv::Scalar(0,255,0), 2);

    //  绘制四个角点 
    for (auto& p : point) {
        cv::circle(frame, p, 4, cv::Scalar(0,0,255), -1);
    }

    // 显示置信度
    std::string label = cv::format("%.2f", conf);
    cv::putText(frame, label,
                cv::Point(box.x, box.y - 5),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(0,255,0), 1);
}


std::vector<yolo_kpt::Object> yolo_kpt::work(cv::Mat src_img) {

    int img_size = IMG_SIZE;

    std::vector<float> padd;
    cv::Mat input = letter_box(src_img, img_size, img_size, padd);
    cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
    auto data = input_tensor1.data<float>();

    for (int h = 0; h < img_size; h++) {
        for (int w = 0; w < img_size; w++) {
            for (int c = 0; c < 3; c++) {
                int idx = c * img_size * img_size + h * img_size + w;
                data[idx] = input.at<cv::Vec3b>(h, w)[c] / 255.0f;
            }
        }
    }

    infer_request.start_async();
    infer_request.wait();

    auto output = infer_request.get_output_tensor(0);
    const float* result = output.data<const float>();
    std::vector<Object> proposals;
    std::vector<Object> results;

    for (auto& obj : proposals) {

        // 坐标还原
        cv::Rect box = scale_box(obj.rect, padd, src_img.cols, src_img.rows);

        Object out;
        out.rect = box;
        out.label = obj.label;
        out.prob = obj.prob;

        // 用 bbox 构造四个角点（用于PnP）
        float x = box.x;
        float y = box.y;
        float w = box.width;
        float h = box.height;

        std::vector<cv::Point2f> kpts;

        // 左上
        kpts.push_back(cv::Point2f(x, y));

        // 右上
        kpts.push_back(cv::Point2f(x + w, y));

        // 右下
        kpts.push_back(cv::Point2f(x + w, y + h));

        // 左下
        kpts.push_back(cv::Point2f(x, y + h));

        out.kpt = kpts;

        results.push_back(out);

#ifdef VIDEO
        drawPred(out.label, out.prob, out.rect, out.kpt, src_img, class_names);
#endif
    }

#ifdef VIDEO
    cv::imshow("result", src_img);
    cv::waitKey(1);
#endif

    return results;
}