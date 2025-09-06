// live_yolov4_tiny_threaded.cpp
// Raspberry Pi (Bookworm) + libcamera + GStreamer + OpenCV DNN (YOLOv4-tiny)
// 스레드 분리: 캡처 스레드(Producer) + 추론/표시 스레드(Consumer)
// - 캡처는 최신 프레임만 유지 (queue size = 1)
// - appsink drop=true sync=false 로 저지연 유지
// - 화면 좌상단에 raw/nms/ms/FPS 를 오버레이하여 추론 동작 확인
// - 키: d(디버그 토글), p(스냅샷), q/ESC(종료)

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

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
static std::string rpi_gst_pipeline_safe(int width, int height, int fps) {
    std::ostringstream ss;
    ss  << "libcamerasrc ! "
        << "video/x-raw,width=" << width << ",height=" << height
        << ",framerate=" << fps << "/1 ! "
        << "videoconvert ! video/x-raw,format=BGR ! "
        << "appsink max-buffers=1 drop=true sync=false";
    return ss.str();
}
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

struct SharedFrame {
    cv::Mat latest;                 // 최신 프레임 (BGR)
    std::mutex m;
    std::condition_variable cv;
    bool has_new = false;           // 새 프레임 도착 플래그
    std::atomic<bool> stop{false};
};

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_INFO);
    // (선택) OpenCV 내부 쓰레드 수 설정
    // cv::setNumThreads(std::thread::hardware_concurrency());

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

    std::vector<std::string> outNames = net.getUnconnectedOutLayersNames();
    if (outNames.empty()) {
        std::cerr << "[ERR] 출력 레이어 이름을 가져오지 못했습니다." << std::endl;
        return -1;
    }
    std::cerr << "[INFO] outNames:";
    for (auto &n : outNames) std::cerr << " " << n;
    std::cerr << std::endl;

    // 4) 카메라 열기
    int camWidth = 640;
    int camHeight = 480;
    int camFps = 30;

    cv::VideoCapture cap;
#if USE_GSTREAMER
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
    if (!cap.open(0, cv::CAP_AVFOUNDATION)) { //CAP_V4L2 나중 웹캠용
        std::cerr << "[ERR] /dev/video0 오픈 실패(V4L2)." << std::endl;
        return -1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, camWidth);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, camHeight);
    cap.set(cv::CAP_PROP_FPS, camFps);
#endif

    if (!cap.isOpened()) {
        std::cerr << "[ERR] 카메라를 열 수 없습니다." << std::endl;
        return -1;
    }

    // 5) 하이퍼파라미터
    const float confThreshold = 0.5f;
    const float nmsThreshold  = 0.4f;
    cv::Size inputSize(416, 416); // 320x320으로 낮추면 FPS 상승

    // 공유 버퍼 및 캡처 스레드 시작
    SharedFrame shared;

    std::thread captureThread([&]() {
        cv::Mat frame;
        while (!shared.stop.load()) {
            if (!cap.read(frame) || frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            if (frame.channels() == 4) {
                cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
            }
            // 최신 프레임만 유지 (덮어쓰기)
            {
                std::lock_guard<std::mutex> lk(shared.m);
                frame.copyTo(shared.latest);
                shared.has_new = true;
            }
            shared.cv.notify_one();
        }
    });

    // 워밍업 1회: 최신 프레임이 들어올 때까지 대기 후 수행
    {
        cv::Mat warm;
        {
            std::unique_lock<std::mutex> lk(shared.m);
            shared.cv.wait(lk, [&]{ return shared.stop.load() || shared.has_new; });
            if (shared.stop.load()) { /* 종료 요청 */ }
            if (!shared.latest.empty()) shared.latest.copyTo(warm);
            shared.has_new = false; // 소비 표시
        }
        if (!warm.empty()) {
            cv::Mat wblob;
            cv::dnn::blobFromImage(warm, wblob, 1.0/255.0, inputSize, cv::Scalar(), true, false);
            net.setInput(wblob);
            std::vector<cv::Mat> wouts; (void)wouts; // 미사용 경고 방지
            net.forward(wouts, outNames);
            std::cerr << "[INFO] 워밍업 1회 완료\n";
        }
    }

    cv::namedWindow("YOLOv4-tiny (Live)", cv::WINDOW_NORMAL);

    bool verbose = false;
    while (true) {
        // 최신 프레임 가져오기 (없으면 대기, 오래된 프레임은 버림)
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lk(shared.m);
            shared.cv.wait(lk, [&]{ return shared.stop.load() || shared.has_new; });
            if (shared.stop.load()) break;
            shared.latest.copyTo(frame);
            shared.has_new = false; // 소비 완료
        }
        if (frame.empty()) continue;

        auto t0 = std::chrono::high_resolution_clock::now();

        // 전처리
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0/255.0, inputSize, cv::Scalar(), true, false);
        net.setInput(blob);

        // 추론
        std::vector<cv::Mat> outs;
        net.forward(outs, outNames);

        // 결과 파싱
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        int rawDet = 0;

        for (size_t i = 0; i < outs.size(); ++i) {
            if (outs[i].total() == 0) continue;
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
                rawDet++;
                float boxScore = data[4];
                if (boxScore < confThreshold) continue;

                cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                cv::Point classIdPoint; double classConf;
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

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
        int nmsDet = (int)indices.size();

        for (int idx : indices) {
            int classId = classIds[idx];
            float confidence = confidences[idx];
            cv::Rect box = boxes[idx];
            cv::Scalar color = colorForClass(classId);
            cv::rectangle(frame, box, color, 2);
            std::string label = (classId >= 0 && classId < (int)classNames.size()) ? classNames[classId] : ("id=" + std::to_string(classId));
            std::ostringstream oss; oss << label << " " << std::fixed << std::setprecision(2) << confidence;
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

        std::ostringstream status;
        status << "raw:" << rawDet << " nms:" << nmsDet
               << " | " << std::fixed << std::setprecision(1) << infer_ms << " ms ("
               << fps << " FPS)";
        cv::putText(frame, status.str(), {10, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    {0,255,0}, 2);

        cv::imshow("YOLOv4-tiny (Live)", frame);
        int key = cv::waitKey(1);
        if (key == 'd') {
            verbose = !verbose;
        } else if (key == 'p') {
            cv::imwrite("snapshot.jpg", frame);
            std::cerr << "[INFO] snapshot saved: snapshot.jpg\n";
        } else if (key == 'q' || key == 27) {
            break;
        }
        if (verbose) {
            std::cerr << "[DBG] raw="<<rawDet<<" nms="<<nmsDet
                      << " time="<<infer_ms<<"ms, FPS="<<fps << std::endl;
        }
    }

    // 종료 처리
    shared.stop.store(true);
    shared.cv.notify_all();
    if (captureThread.joinable()) captureThread.join();

    return 0;
}
