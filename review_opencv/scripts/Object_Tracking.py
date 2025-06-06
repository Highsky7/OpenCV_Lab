import os
import sys
import cv2
import numpy as np

# --- 헬퍼 함수 (수정 없음) ---
def drawRectangle(frame, bbox):
    """프레임에 사각형을 그립니다."""
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

def drawText(frame, txt, location, color=(50, 170, 50)):
    """프레임에 텍스트를 씁니다."""
    cv2.putText(frame, txt, location, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

# --- 메인 실행 로직 ---

# 입력 비디오 파일 경로
video_input_file_name = "review_opencv/images/race_car.mp4"

# 실행할 모든 트래커 유형 리스트
tracker_types = [
    "BOOSTING",
    "MIL",
    "KCF",
    "CSRT",
    "TLD",
    "MEDIANFLOW",
    "GOTURN",
    "MOSSE",
]

# 추적할 대상의 초기 위치 (x, y, width, height)
initial_bbox = (1300, 405, 160, 120)


# tracker_types 리스트의 모든 트래커를 순회하는 반복문
for tracker_type in tracker_types:

    # 현재 어떤 트래커를 실행 중인지 터미널에 출력
    print(f"--- 현재 처리 중인 트래커: {tracker_type} ---")

    # 트래커 객체 생성
    if tracker_type == "BOOSTING":
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == "MIL":
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == "KCF":
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    elif tracker_type == "TLD":
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == "MEDIANFLOW":
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_type == "GOTURN":
        # GOTURN 모델 파일 경로 지정
        model_txt = "review_opencv/images/goturn.prototxt"
        model_bin = "review_opencv/images/goturn.caffemodel"

        # 모델 파일이 실제로 존재하는지 확인
        if not os.path.exists(model_txt) or not os.path.exists(model_bin):
            print(f"오류: GOTURN 모델 파일이 지정된 경로에 없습니다. 건너뜁니다.")
            print(f"필요한 파일: {model_txt}, {model_bin}")
            continue # 파일이 없으면 다음 트래커로 넘어감
        
        # GOTURN 트래커 생성
        # OpenCV 최신 버전에서는 파라미터 전달 방식이 다를 수 있으나,
        # create() 함수가 직접 파일 경로를 인자로 받는 경우가 일반적입니다.
        try:
            tracker = cv2.TrackerGOTURN.create(model_txt, model_bin)
        except AttributeError:
             # 구버전 OpenCV와의 호환성을 위해 에러 발생 시 안내 메시지 출력
            print("오류: 사용 중인 OpenCV 버전에서는 이 방식으로 GOTURN을 생성할 수 없습니다. 건너뜁니다.")
            continue
    else: # MOSSE
        tracker = cv2.legacy.TrackerMOSSE_create()


    # 비디오 파일을 새로 엽니다. (각 트래커마다 처음부터 다시 읽어야 함)
    video = cv2.VideoCapture(video_input_file_name)
    if not video.isOpened():
        print(f"오류: 비디오 파일({video_input_file_name})을 열 수 없습니다.")
        continue  # 다음 트래커로 넘어감

    # 비디오의 너비와 높이 정보 가져오기
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 각 트래커의 이름이 포함된 결과 비디오 파일 설정
    video_output_file_name = f"review_opencv/images/race_car-{tracker_type}.mp4"
    video_out = cv2.VideoWriter(video_output_file_name, cv2.VideoWriter_fourcc(*"mp4v"), 25, (width, height))

    # 첫 번째 프레임 읽기
    ok, frame = video.read()
    if not ok:
        print(f"오류: 첫 프레임을 읽을 수 없습니다.")
        video.release()
        video_out.release()
        continue

    # 읽어온 첫 프레임과 바운딩 박스로 트래커 초기화
    try:
        ok = tracker.init(frame, initial_bbox)
        if not ok:
            print(f"오류: {tracker_type} 트래커 초기화에 실패했습니다.")
            video.release()
            video_out.release()
            continue
    except Exception as e:
        print(f"{tracker_type} 트래커 초기화 중 에러 발생: {e}")
        video.release()
        video_out.release()
        continue


    # 비디오의 끝까지 프레임 단위로 루프 실행
    while True:
        ok, frame = video.read()
        if not ok:
            break

        timer = cv2.getTickCount()
        ok, bbox = tracker.update(frame)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        if ok:
            drawRectangle(frame, bbox)
        else:
            drawText(frame, "Tracking failure detected", (80, 140), (0, 0, 255))

        drawText(frame, tracker_type + " Tracker", (80, 60))
        drawText(frame, "FPS : " + str(int(fps)), (80, 100))

        video_out.write(frame)

    video.release()
    video_out.release()

    print(f"--- {tracker_type} 트래커 처리 완료. 결과 저장: {video_output_file_name} ---")

print("\n>>> 모든 트래커 처리가 완료되었습니다.")