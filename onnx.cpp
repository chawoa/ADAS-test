// rpi_yolo_onnx_threaded.cpp
// Raspberry Pi (Bookworm) + libcamera + GStreamer + OpenCV DNN (ONNX: YOLOv5n/YOLOv8n only)
// 3-스레드 구조: 캡처(Producer) / 추론(Consumer) / 렌더(Consumer)
// - 캡처: 최신 프레임만 유지 (queue size=1)
// - 추론: 최신 프레임만 받아 전처리+forward, 결과를 렌더 큐로 전달
// - 렌더: 그리기/표시/키 입력 처리. q/ESC 종료, p 스냅샷, d 디버그 토글
// - 좌상단 오버레이: raw/nms/ms/FPS

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

// ======= 환경 설정 =======
// GStreamer(libcamerasrc)만 사용 (라즈베리파이 전용)
#define USE_GSTREAMER 1

// ONNX 모델 경로(예: YOLOv5n 또는 YOLOv8n)
static std::string kOnnxModelPath = "model/yolov8n.onnx"; // 필요시 "model/yolov8n.onnx"로 변경
static std::string kClassNamesPath = "model/coco.names";  // COCO 80 classes

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

struct LatestBuffer {
    cv::Mat frame;                  // 최신 캡처 프레임(BGR)
    bool has_new = false;           // 새 프레임 도착
    std::mutex m;                   // 보호용 뮤텍스
    std::condition_variable cv;     // 대기/알림
};

struct DisplayBuffer {
    cv::Mat frame;                  // 렌더링용 프레임(오버레이 포함)
    bool has_new = false;
    std::mutex m;
    std::condition_variable cv;
};

// ===== ONNX(YOLOv5/YOLOv8) 출력 파싱 유틸 =====
// 다양한 ONNX 내보내기 형태를 수용:
// - (1, N, no) : no = 5+nc (v5), 또는 4+nc (v8: obj 없음)
// - (1, no, N) : 전치 형태 (v8에서 흔함)
// - (N, no)    : 이미 2D
static void parseYoloONNX(const cv::Mat &rawOut,
                          int imgW, int imgH, float confThreshold,
                          std::vector<int> &classIds,
                          std::vector<float> &confidences,
                          std::vector<cv::Rect> &boxes,
                          int &rawDet)
{
    rawDet = 0;

    // NxNo 형태로 맞추기
    cv::Mat out2d; // rows=N, cols=no
    if (rawOut.dims == 3 && rawOut.size[0] == 1) {
        int a = rawOut.size[1];
        int b = rawOut.size[2];
        const float* base = rawOut.ptr<float>();
        if (b >= 6 && a > b) {
            // (1, N, no)
            out2d = cv::Mat(a, b, CV_32F, (void*)base);
        } else if (a >= 6 && b > a) {
            // (1, no, N) → 전치해서 (N, no)
            cv::Mat noxN(a, b, CV_32F, (void*)base);
            cv::transpose(noxN, out2d);
        } else {
            // 모호: 기본 가정으로 (1, N, no)
            out2d = cv::Mat(a, b, CV_32F, (void*)base);
        }
    } else if (rawOut.dims == 2) {
        out2d = rawOut; // 이미 2D
    } else {
        // 알 수 없는 형태
        return;
    }

    const int N  = out2d.rows;
    const int no = out2d.cols; // no = 5+nc (v5) 또는 4+nc (v8)

    bool hasObjectness = (no - 5) >= 1; // 85같이 5+nc 구조면 true, 84(4+80)면 false
    int  clsStart      = hasObjectness ? 5 : 4;
    int  numClasses    = no - clsStart;

    for (int i = 0; i < N; ++i) {
        const float* p = out2d.ptr<float>(i);
        rawDet++;

        float cx = p[0] * imgW;
        float cy = p[1] * imgH;
        float w  = p[2] * imgW;
        float h  = p[3] * imgH;

        float obj = hasObjectness ? p[4] : 1.0f;
        if (obj < 1e-6f) continue; // 안전 하한

        cv::Mat scores(1, numClasses, CV_32F, (void*)(p + clsStart));
        cv::Point classIdPoint; double classConf;
        cv::minMaxLoc(scores, 0, &classConf, 0, &classIdPoint);

        float confidence = (float)classConf * obj; // v8(no obj)일 땐 obj=1
        if (confidence < confThreshold) continue;

        int left = (int)(cx - w * 0.5f);
        int top  = (int)(cy - h * 0.5f);
        int iw   = (int)w;
        int ih   = (int)h;

        left = std::max(0, left);
        top  = std::max(0, top);
        iw   = std::min(iw, imgW - left);
        ih   = std::min(ih, imgH - top);

        classIds.push_back(classIdPoint.x);
        confidences.push_back(confidence);
        boxes.emplace_back(left, top, iw, ih);
    }
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_INFO);

    // 1) 클래스 이름 로드 (없어도 동작)
    std::vector<std::string> classNames;
    {
        std::ifstream ifs(kClassNamesPath.c_str());
        if (!ifs.is_open()) {
            std::cerr << "[WARN] " << kClassNamesPath << " 를 열 수 없습니다. 라벨은 id로만 표시합니다.\n";
        } else {
            std::string line;
            while (std::getline(ifs, line)) classNames.push_back(line);
            if (classNames.empty()) {
                std::cerr << "[WARN] 클래스 이름이 비어 있습니다.\n";
            }
        }
    }

    // 2) ONNX 네트워크 로드
    cv::dnn::Net net = cv::dnn::readNetFromONNX(kOnnxModelPath);
    if (net.empty()) {
        std::cerr << "[ERR] ONNX 모델 로드 실패: " << kOnnxModelPath << "\n";
        return -1;
    }
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    std::cerr << "[INFO] Using ONNX model: " << kOnnxModelPath << "\n";

    // 3) 카메라 열기 (GStreamer + libcamerasrc)
    int camWidth = 640;    // 성능 고려(필요시 1280x720)
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
            std::cerr << "      파이프라인: " << pipeline << "\n";
            return -1;
        }
    }
    std::cerr << "[INFO] GStreamer pipeline: " << pipeline << "\n";
#else
    #error "This file is Raspberry Pi + GStreamer only."
#endif

    if (!cap.isOpened()) {
        std::cerr << "[ERR] 카메라를 열 수 없습니다.\n";
        return -1;
    }

    // 4) 하이퍼파라미터
    const float confThreshold = 0.5f;
    const float nmsThreshold  = 0.4f;
    cv::Size inputSize(640, 640);   // 경량 ONNX: 320 권장 (속도↑)

    std::atomic<bool> stop{false};
    LatestBuffer latest;            // 캡처→추론 전달 버퍼
    DisplayBuffer display;          // 추론→렌더 전달 버퍼

    // --- 캡처 스레드 ---
    std::thread captureThread([&](){
        cv::Mat frame;
        while (!stop.load()) {
            if (!cap.read(frame) || frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            if (frame.channels() == 4) {
                cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
            }
            {
                std::lock_guard<std::mutex> lk(latest.m);
                frame.copyTo(latest.frame);   // 최신 프레임으로 교체
                latest.has_new = true;
            }
            latest.cv.notify_one();
        }
    });

    // --- 렌더 스레드 ---
    bool verbose = false;
    std::thread renderThread([&](){
        cv::namedWindow("YOLO (Live)", cv::WINDOW_NORMAL);
        while (!stop.load()) {
            cv::Mat show;
            {
                std::unique_lock<std::mutex> lk(display.m);
                display.cv.wait(lk, [&]{ return stop.load() || display.has_new; });
                if (stop.load()) break;
                display.frame.copyTo(show);
                display.has_new = false; // 소비
            }
            if (show.empty()) continue;
            cv::imshow("YOLO (Live)", show);
            int key = cv::waitKey(1);
            if (key == 'd') {
                verbose = !verbose;
            } else if (key == 'p') {
                cv::imwrite("snapshot.jpg", show);
                std::cerr << "[INFO] snapshot saved: snapshot.jpg\n";
            } else if (key == 'q' || key == 27) {
                stop.store(true);
                latest.cv.notify_all();
                display.cv.notify_all();
                break;
            }
        }
    });

    // --- 추론(메인) 스레드 ---
    // 워밍업: 첫 프레임 대기 → 1회 forward
    {
        cv::Mat warm;
        std::unique_lock<std::mutex> lk(latest.m);
        latest.cv.wait(lk, [&]{ return stop.load() || latest.has_new; });
        if (!stop.load() && !latest.frame.empty()) {
            latest.frame.copyTo(warm);
            latest.has_new = false;
        }
        lk.unlock();
        if (!warm.empty()) {
            cv::Mat wblob;
            cv::dnn::blobFromImage(warm, wblob, 1.0/255.0, inputSize, cv::Scalar(), true, false);
            net.setInput(wblob);
            std::vector<cv::Mat> wouts; net.forward(wouts);
            std::cerr << "[INFO] 워밍업 1회 완료\n";
        }
    }

    while (!stop.load()) {
        // 최신 프레임 받기
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lk(latest.m);
            latest.cv.wait(lk, [&]{ return stop.load() || latest.has_new; });
            if (stop.load()) break;
            latest.frame.copyTo(frame);
            latest.has_new = false;
        }
        if (frame.empty()) continue;

        auto t0 = std::chrono::high_resolution_clock::now();

        // 전처리 (swapRB=true)
        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0/255.0, inputSize, cv::Scalar(), true, false);
        net.setInput(blob);

        // forward & 파싱
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        int rawDet = 0;

        std::vector<cv::Mat> outs; net.forward(outs);
        if (!outs.empty()) {
            parseYoloONNX(outs[0], frame.cols, frame.rows, confThreshold,
                          classIds, confidences, boxes, rawDet);
        }

        // NMS 및 그리기
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
        int nmsDet = (int)indices.size();

        cv::Mat show = frame.clone();
        for (int idx : indices) {
            int classId = classIds[idx];
            float confidence = confidences[idx];
            cv::Rect box = boxes[idx];
            cv::Scalar color = colorForClass(classId);
            cv::rectangle(show, box, color, 2);
            std::string label = (classId >= 0 && classId < (int)classNames.size()) ? classNames[classId] : ("id=" + std::to_string(classId));
            std::ostringstream oss; oss << label << " " << std::fixed << std::setprecision(2) << confidence;
            int baseLine = 0;
            cv::Size labelSize = cv::getTextSize(oss.str(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            int top = std::max(box.y, labelSize.height + 10);
            cv::rectangle(show,
                          cv::Point(box.x, top - labelSize.height - 8),
                          cv::Point(box.x + labelSize.width + 8, top),
                          color, cv::FILLED);
            cv::putText(show, oss.str(),
                        cv::Point(box.x + 4, top - 4),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(255,255,255), 1);
            cv::circle(show, (box.br() + box.tl()) * 0.5, 3, color, cv::FILLED);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double fps = 1000.0 / std::max(1.0, infer_ms);

        std::ostringstream status;
        status << "raw:" << rawDet << " nms:" << nmsDet
               << " | " << std::fixed << std::setprecision(1) << infer_ms << " ms ("
               << fps << " FPS)";
        cv::putText(show, status.str(), {10, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    {0,255,0}, 2);

        // 렌더 큐에 최신 프레임으로 교체
        {
            std::lock_guard<std::mutex> lk(display.m);
            show.copyTo(display.frame);
            display.has_new = true;
        }
        display.cv.notify_one();

        if (verbose) {
            std::cerr << "[DBG] raw="<<rawDet<<" nms="<<nmsDet
                      << " time="<<infer_ms<<"ms, FPS="<<fps << "\n";
        }
    }

    // 종료 처리
    stop.store(true);
    latest.cv.notify_all();
    display.cv.notify_all();

    // 스레드 조인
    if (captureThread.joinable()) captureThread.join();
    if (renderThread.joinable())  renderThread.join();

    return 0;
}
