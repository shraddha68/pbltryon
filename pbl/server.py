from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)
selected_item = None

camera = cv2.VideoCapture(0)  # use cv2.CAP_DSHOW on Windows if needed
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Smooth landmark positions
smoothed_left_shoulder = None
smoothed_right_shoulder = None
smoothed_left_hip = None
SMOOTHING_FACTOR = 0.2  # controls how smooth the cloth moves


def load_cloth_overlay(path):
    overlay = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return overlay


def overlay_cloth(frame, overlay_img, landmarks):
    global smoothed_left_shoulder, smoothed_right_shoulder, smoothed_left_hip

    if overlay_img is None or landmarks is None:
        return frame

    try:
        h, w, _ = frame.shape

        # Get raw landmark points
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

        # Convert to pixel coordinates
        curr_l_sh = np.array([int(left_shoulder.x * w), int(left_shoulder.y * h)])
        curr_r_sh = np.array([int(right_shoulder.x * w), int(right_shoulder.y * h)])
        curr_l_hip = np.array([int(left_hip.x * w), int(left_hip.y * h)])

        # Smooth movement
        if smoothed_left_shoulder is None:
            smoothed_left_shoulder = curr_l_sh
            smoothed_right_shoulder = curr_r_sh
            smoothed_left_hip = curr_l_hip
        else:
            smoothed_left_shoulder = (1 - SMOOTHING_FACTOR) * smoothed_left_shoulder + SMOOTHING_FACTOR * curr_l_sh
            smoothed_right_shoulder = (1 - SMOOTHING_FACTOR) * smoothed_right_shoulder + SMOOTHING_FACTOR * curr_r_sh
            smoothed_left_hip = (1 - SMOOTHING_FACTOR) * smoothed_left_hip + SMOOTHING_FACTOR * curr_l_hip

        l_sh = smoothed_left_shoulder.astype(int)
        r_sh = smoothed_right_shoulder.astype(int)
        l_hip = smoothed_left_hip.astype(int)

        # Width and height
        shoulder_dist = int(np.linalg.norm(r_sh - l_sh))
        cloth_width = int(shoulder_dist * 1.5)
        cloth_height = int(abs(l_hip[1] - l_sh[1]) * 1.2)

        center_x = int((l_sh[0] + r_sh[0]) / 2)
        top_y = int(min(l_sh[1], r_sh[1]) - 20)

        resized_overlay = cv2.resize(overlay_img, (cloth_width, cloth_height), interpolation=cv2.INTER_AREA)

        x1 = center_x - cloth_width // 2
        y1 = top_y
        x2 = x1 + cloth_width
        y2 = y1 + cloth_height

        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return frame

        alpha_s = resized_overlay[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(3):
            frame[y1:y2, x1:x2, c] = (
                alpha_s * resized_overlay[:, :, c] +
                alpha_l * frame[y1:y2, x1:x2, c]
            )

    except Exception as e:
        print("Overlay error:", e)

    return frame


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/camera')
def camera_page():
    return render_template('camera.html')


@app.route('/video_feed')
def video_feed():
    def generate():
        global selected_item
        while True:
            success, frame = camera.read()
            if not success:
                print("Failed to grab frame")
                break

            try:
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if results.pose_landmarks and selected_item:
                    landmarks = results.pose_landmarks.landmark
                    overlay_path = f"static/assets/{selected_item}"
                    overlay_img = load_cloth_overlay(overlay_path)
                    frame = overlay_cloth(frame, overlay_img, landmarks)

                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue

                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       buffer.tobytes() + b'\r\n')
            except Exception as e:
                print("Frame error:", e)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/select-item', methods=['POST'])
def select_item():
    global selected_item
    data = request.get_json()
    selected_item = data['item']
    print("Selected item:", selected_item)
    return jsonify(status='success')


if __name__ == "__main__":
    app.run(debug=True)
