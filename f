import cv2
import mediapipe as mp
from controller import Robot, Keyboard, Motion

class Nao(Robot):
    PHALANX_MAX = 8

    def __init__(self):
        Robot.__init__(self)
        self.currentlyPlaying = False
        self.findAndEnableDevices()
        self.loadMotionFiles()

        # Initialize Mediapipe for hand and pose tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = cv2.VideoCapture(0)

    def loadMotionFiles(self):
        # Load motion files (ensure paths are correct)
        self.forwards = Motion('../../motions/Forwards50.motion')
        self.backwards = Motion('../../motions/Backwards.motion')
        self.sideStepLeft = Motion('../../motions/SideStepLeft.motion')
        self.sideStepRight = Motion('../../motions/SideStepRight.motion')
        self.turnLeft60 = Motion('../../motions/TurnLeft60.motion')
        self.turnRight60 = Motion('../../motions/TurnRight60.motion')
        self.taiChi = Motion('../../motions/TaiChi.motion')
        self.wipeForhead = Motion('../../motions/WipeForehead.motion')

    def findAndEnableDevices(self):
        self.timeStep = int(self.getBasicTimeStep())

        # Enable devices like camera, accelerometer, gyro, etc.
        self.cameraTop = self.getDevice("CameraTop")
        self.cameraBottom = self.getDevice("CameraBottom")
        self.cameraTop.enable(4 * self.timeStep)
        self.cameraBottom.enable(4 * self.timeStep)

        self.accelerometer = self.getDevice('accelerometer')
        self.accelerometer.enable(4 * self.timeStep)

        self.gyro = self.getDevice('gyro')
        self.gyro.enable(4 * self.timeStep)

        self.gps = self.getDevice('gps')
        self.gps.enable(4 * self.timeStep)

        self.inertialUnit = self.getDevice('inertial unit')
        self.inertialUnit.enable(self.timeStep)

        self.us = []
        usNames = ['Sonar/Left', 'Sonar/Right']
        for name in usNames:
            sensor = self.getDevice(name)
            sensor.enable(self.timeStep)
            self.us.append(sensor)

        self.fsr = []
        fsrNames = ['LFsr', 'RFsr']
        for name in fsrNames:
            sensor = self.getDevice(name)
            sensor.enable(self.timeStep)
            self.fsr.append(sensor)

        self.lfootlbumper = self.getDevice('LFoot/Bumper/Left')
        self.lfootrbumper = self.getDevice('LFoot/Bumper/Right')
        self.rfootlbumper = self.getDevice('RFoot/Bumper/Left')
        self.rfootrbumper = self.getDevice('RFoot/Bumper/Right')
        self.lfootlbumper.enable(self.timeStep)
        self.lfootrbumper.enable(self.timeStep)
        self.rfootlbumper.enable(self.timeStep)
        self.rfootrbumper.enable(self.timeStep)

        self.leds = [
            self.getDevice('ChestBoard/Led'),
            self.getDevice('RFoot/Led'),
            self.getDevice('LFoot/Led'),
            self.getDevice('Face/Led/Right'),
            self.getDevice('Face/Led/Left'),
            self.getDevice('Ears/Led/Right'),
            self.getDevice('Ears/Led/Left')
        ]

        self.lphalanx = []
        self.rphalanx = []
        self.maxPhalanxMotorPosition = []
        self.minPhalanxMotorPosition = []
        for i in range(self.PHALANX_MAX):
            l_motor = self.getDevice(f"LPhalanx{i + 1}")
            r_motor = self.getDevice(f"RPhalanx{i + 1}")
            self.lphalanx.append(l_motor)
            self.rphalanx.append(r_motor)

            if r_motor:
                self.maxPhalanxMotorPosition.append(r_motor.getMaxPosition())
                self.minPhalanxMotorPosition.append(r_motor.getMinPosition())

        self.RShoulderPitch = self.getDevice("RShoulderPitch")
        self.LShoulderPitch = self.getDevice("LShoulderPitch")

        self.keyboard = self.getKeyboard()
        self.keyboard.enable(10 * self.timeStep)

    def detectHandGestures(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_hands = self.hands.process(frame_rgb)
        result_pose = self.pose.process(frame_rgb)

        if result_hands.multi_hand_landmarks and result_hands.multi_handedness:
            for hand_landmarks, handedness in zip(result_hands.multi_hand_landmarks, result_hands.multi_handedness):
                hand_label = handedness.classification[0].label  # "Left" or "Right"

                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Определение закрытия или открытия ладони для каждой руки отдельно
                if thumb_tip.y > index_tip.y:  # Открытая ладонь
                    if hand_label == 'Left':
                        self.setLeftHandAngle(0.96)
                    else:
                        self.setRightHandAngle(0.96)
                else:  # Закрытая ладонь
                    if hand_label == 'Left':
                        self.setLeftHandAngle(0.0)
                    else:
                        self.setRightHandAngle(0.0)

                # Управление движением плеч для каждой руки
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                shoulder_angle = (wrist.y - middle_finger_tip.y) * 3  # Scaling factor
                shoulder_angle = max(min(shoulder_angle, self.RShoulderPitch.getMaxPosition()), self.RShoulderPitch.getMinPosition())

                if hand_label == 'Left':
                    self.LShoulderPitch.setPosition(shoulder_angle)
                else:
                    self.RShoulderPitch.setPosition(shoulder_angle)

                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        if result_pose.pose_landmarks:
            # Здесь можно добавить логику для управления другими частями тела, если поддерживается
            self.mp_draw.draw_landmarks(frame, result_pose.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Pose and Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()

    def setLeftHandAngle(self, angle):
        for i in range(self.PHALANX_MAX):
            clampedAngle = max(min(angle, self.maxPhalanxMotorPosition[i]), self.minPhalanxMotorPosition[i])
            if len(self.lphalanx) > i and self.lphalanx[i] is not None:
                self.lphalanx[i].setPosition(clampedAngle)

    def setRightHandAngle(self, angle):
        for i in range(self.PHALANX_MAX):
            clampedAngle = max(min(angle, self.maxPhalanxMotorPosition[i]), self.minPhalanxMotorPosition[i])
            if len(self.rphalanx) > i and self.rphalanx[i] is not None:
                self.rphalanx[i].setPosition(clampedAngle)

    def run(self):
        while robot.step(self.timeStep) != -1:
            self.detectHandGestures()  # Continuously check for hand and pose gestures
            key = self.keyboard.getKey()
            if key > 0:
                break

            if robot.step(self.timeStep) == -1:
                break

# Create the Robot instance and run main loop
robot = Nao()
robot.run()
