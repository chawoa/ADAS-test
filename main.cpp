// live_yolov4_tiny.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <chrono>
#include <sstream>

#include <opencv2/core/utils/logger.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

static cv::Scalar colorForClass(int classId) {
    uint32_t h = (uint32_t)classId * 2654435761u;
    int b = (h & 0xFF);
    int g = ((h >> 8) & 0xFF);
    int r = ((h >> 16) & 0xFF);
    return cv::Scalar(b, g, r);
}

// ▼ 필요 시 0으로 바꾸시면 V4L2(/dev/video0) 사용
#define USE_GSTREAMER 1

#if USE_GSTREAMER
// 안전 버전: 포맷 미지정(협상), BGR로 변환 후 appsink
static std::string rpi_gst_pipeline_safe(int width, int height, int fps) {
    std::ostringstream ss;
    ss  << "libcamerasrc ! "
        << "video/x-raw,width=" << width << ",height=" << height
        << ",framerate=" << fps << "/1 ! "
        << "videoconvert ! video/x-raw,format=BGR ! "
        << "appsink max-buffers=1 drop=true sync=false";
    return ss.str();
}

// 명시 버전: NV12로 요청 → BGR 변환 → appsink
static std::string rpi_gst_pipeline_nv12(int width, int height, int fps) {
    std::ostringstream ss;
    ss  << "libcamerasrc ! "
        << "video/x-raw,format=NV12,width=" << width << ",height=" << height
        << ",framerate=" << fps << "/1 ! "
        << "videoconvert ! video/x-raw,format=BGR ! "
        << "appsink max-buffers=1 drop=true sync=false";
    return ss.str();
}
#endif

int main() {
    // OpenCV 내부 로그(필요 시 INFO로)
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_INFO);

    // 1) 모델/클래스 경로
    std::string modelWeights = "model/yolov4-tiny.weights";
    std::string modelConfig  = "model/yolov4-tiny.cfg";
    std::string classNamesFile = "model/coco.names";

    // 2) 클래스 이름 로드
    std::vector<std::string> classNames;
    {
        std::ifstream ifs(classNamesFile.c_str());
        if (!ifs.is_open()) {
            std::cerr << "[ERR] 클래스 파일을 열 수 없습니다: " << classNamesFile << std::endl;
            return -1;
        }
        std::string line;
        while (std::getline(ifs, line)) classNames.push_back(line);
        if (classNames.empty()) {
            std::cerr << "[WARN] 클래스 이름이 비어 있습니다." << std::endl;
        }
    }

    // 3) 네트워크 로드
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(modelConfig, modelWeights);
    if (net.empty()) {
        std::cerr << "[ERR] Darknet 모델 로드 실패: cfg=" << modelConfig
                  << " weights=" << modelWeights << std::endl;
        return -1;
    }
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // 출력 레이어 이름 확인
    std::vector<std::string> outNames = net.getUnconnectedOutLayersNames();
    if (outNames.empty()) {
        std::cerr << "[ERR] 출력 레이어 이름을 가져오지 못했습니다." << std::endl;
        return -1;
    }
    std::cerr << "[INFO] outNames:";
    for (auto &n : outNames) std::cerr << " " << n;
    std::cerr << std::endl;

    // 4) 카메라 열기
    int camWidth = 640;    // 성능 고려(원하시면 1280x720)
    int camHeight = 480;
    int camFps = 30;

    cv::VideoCapture cap;
#if USE_GSTREAMER
    // 우선 안전 파이프라인으로 시도 → 실패하면 NV12 버전 시도
    std::string pipeline = rpi_gst_pipeline_safe(camWidth, camHeight, camFps);
    if (!cap.open(pipeline, cv::CAP_GSTREAMER)) {
        std::cerr << "[WARN] 안전 파이프라인 오픈 실패, NV12 버전 재시도\n";
        pipeline = rpi_gst_pipeline_nv12(camWidth, camHeight, camFps);
        if (!cap.open(pipeline, cv::CAP_GSTREAMER)) {
            std::cerr << "[ERR] GStreamer 파이프라인 오픈 실패.\n";
            std::cerr << "      파이프라인: " << pipeline << std::endl;
            return -1;
        }
    }
    std::cerr << "[INFO] GStreamer pipeline: " << pipeline << std::endl;
#else
    // /dev/video0 (V4L2) — libcamera v4l2 호환 또는 USB 웹캠
    if (!cap.open(0, cv::CAP_V4L2)) {
        std::cerr << "[ERR] /dev/video0 오픈 실패(V4L2).\n";
        return -1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, camWidth);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, camHeight);
    cap.set(cv::CAP_PROP_FPS, camFps);
#endif

    if (!cap.isOpened()) {
        std::cerr << "[ERR] 카메라를 열 수 없습니다. (GStreamer 또는 V4L2 경로 확인)" << std::endl;
        return -1;
    }

    // 5) 추론 하이퍼파라미터
    const float confThreshold = 0.5f;
    const float nmsThreshold  = 0.4f;
    const cv::Size inputSize(416, 416); // 320x320으로 낮추면 FPS↑, 정확도는 약간↓

    // === 워밍업(선택) ===
    {
        cv::Mat warm;
        if (cap.read(warm) && !warm.empty()) {
            if (warm.channels() == 4) {
                cv::cvtColor(warm, warm, cv::COLOR_BGRA2BGR);
            }
            cv::Mat wblob;
            cv::dnn::blobFromImage(warm, wblob, 1.0/255.0, inputSize, cv::Scalar(), true, false);
            net.setInput(wblob);
            std::vector<cv::Mat> wouts;
            net.forward(wouts, outNames);
            std::cerr << "[INFO] 워밍업 1회 완료\n";
        }
    }

    // 창 생성
    cv::namedWindow("YOLOv4-tiny (Live)", cv::WINDOW_NORMAL);

    bool verbose = false;
    cv::Mat frame;
    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "[ERR] 프레임을 가져올 수 없습니다." << std::endl;
            break;
        }

        // 드물게 BGRx(4채널)로 들어오는 경우 방지
        if (frame.channels() == 4) {
            cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        // 6) 전처리(blob)
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0/255.0, inputSize, cv::Scalar(), true, false);
        net.setInput(blob);

        // 7) 순방향 추론
        std::vector<cv::Mat> outs;
        net.forward(outs, outNames);

        // 8) 결과 파싱
        const int numOuts = static_cast<int>(outs.size());
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        int rawDet = 0;
        for (int i = 0; i < numOuts; ++i) {
            if (outs[i].total() == 0) continue;
            if (outs[i].cols != 85) {
                // COCO(80 classes) 기준 기대값 85(=4+1+80), 다르더라도 경고만
                std::cerr << "[WARN] outs["<<i<<"].cols="<<outs[i].cols<<" (COCO=85 기대)\n";
            }
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
                rawDet++;
                float boxScore = data[4];
                if (boxScore < confThreshold) continue;

                cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                cv::Point classIdPoint;
                double classConf;
                cv::minMaxLoc(scores, 0, &classConf, 0, &classIdPoint);

                float confidence = (float)classConf * boxScore;
                if (confidence > confThreshold) {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width   = (int)(data[2] * frame.cols);
                    int height  = (int)(data[3] * frame.rows);
                    int left    = centerX - width / 2;
                    int top     = centerY - height / 2;

                    left   = std::max(0, left);
                    top    = std::max(0, top);
                    width  = std::min(width,  frame.cols - left);
                    height = std::min(height, frame.rows - top);

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back(confidence);
                    boxes.emplace_back(left, top, width, height);
                }
            }
        }

        // 9) NMS 및 그리기
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
        int nmsDet = (int)indices.size();

        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            int classId = classIds[idx];
            float confidence = confidences[idx];
            std::string label = "id=" + std::to_string(classId);
            if (classId >= 0 && classId < (int)classNames.size())
                label = classNames[classId];

            cv::Rect box = boxes[idx];
            cv::Scalar color = colorForClass(classId);

            cv::rectangle(frame, box, color, 2);

            std::ostringstream oss;
            oss << label << " " << std::fixed << std::setprecision(2) << confidence;

            int baseLine = 0;
            cv::Size labelSize = cv::getTextSize(oss.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            int top = std::max(box.y, labelSize.height + 10);
            cv::rectangle(frame,
                          cv::Point(box.x, top - labelSize.height - 8),
                          cv::Point(box.x + labelSize.width + 8, top),
                          color, cv::FILLED);
            cv::putText(frame, oss.str(),
                        cv::Point(box.x + 4, top - 4),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(255,255,255), 1);

            cv::circle(frame, (box.br() + box.tl()) * 0.5, 3, color, cv::FILLED);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double fps = 1000.0 / std::max(1.0, infer_ms);

        // 상태 오버레이 (raw/nms/ms/FPS)
        std::ostringstream status;
        status << "raw:" << rawDet << " nms:" << nmsDet
               << " | " << std::fixed << std::setprecision(1) << infer_ms << " ms ("
               << fps << " FPS)";
        cv::putText(frame, status.str(), {10, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    {0,255,0}, 2);

        cv::imshow("YOLOv4-tiny (Live)", frame);

        // 키 처리
        int key = cv::waitKey(1);
        if (key == 'd') verbose = !verbose;
        if (verbose) {
            std::cerr << "[DBG] raw="<<rawDet<<" nms="<<nmsDet
                      << " time="<<infer_ms<<"ms, FPS="<<fps << std::endl;
        }
        if (key == 'p') {
            cv::imwrite("snapshot.jpg", frame);
            std::cerr << "[INFO] snapshot saved: snapshot.jpg\n";
        }
        if (key == 'q' || key == 27) break;
    }

    return 0;
}
