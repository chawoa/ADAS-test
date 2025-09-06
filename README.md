자율 주행 테스트
1단계 베이스 Yolo + OpenCV

필요 패키지
sudo apt update
sudo apt install -y libcamera-apps gstreamer1.0-tools \
  gstreamer1.0-libcamera gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad

빌드 명령어 예시: g++ live_yolov4_tiny.cpp -o live_yolo `pkg-config --cflags --libs opencv4`
./live_yolo

맥북에서 사용시
#define USE_GSTREAMER 0
cap.open(0, cv::CAP_AVFOUNDATION)