import cv2, Leap, ctypes, sys
import numpy as np

distances_file = open("./vector_distance.csv", 'a')
distances_file_2 = open("./vector_distance_2.csv", 'a')


def run(controller):
    save_gesture = False

    while True:
        frame = controller.frame()
        image = frame.images[1]
        if image.is_valid:
            i_address = int(image.data_pointer)
            ctype_array_def = ctypes.c_ubyte * image.height * image.width
            # as ctypes array
            as_ctype_array = ctype_array_def.from_address(i_address)
            # as numpy array
            as_numpy_array = np.ctypeslib.as_array(as_ctype_array)
            img = np.reshape(as_numpy_array, (image.height, image.width))

            if save_gesture:
                for hand in frame.hands:
                    v = []
                    dv = []
                    prev_finger = None
                    c = hand.palm_position
                    m = 1
                    for finger in hand.fingers:
                        if prev_finger:
                            d2 = finger.tip_position.distance_to(prev_finger.tip_position)
                            dv.append(d2)
                            prev_finger = finger
                        else:
                            prev_finger = finger
                        d = finger.tip_position.distance_to(c)
                        m = d if finger.type == Leap.Finger.TYPE_MIDDLE else m
                        v.append(d)
                    print(v, dv)
                    np.savetxt(distances_file, [np.append(np.array(v) / m, frame.timestamp)], delimiter=',',
                               fmt=['%f', '%f', '%f', '%f', '%f', '%d'])
                    np.savetxt(distances_file_2, [np.append(np.array(dv) / m, frame.timestamp)], delimiter=',',
                               fmt=['%f', '%f', '%f', '%f', '%d'])
                    cv2.imwrite("./images/%d.png" % frame.timestamp, img)

            cv2.imshow('Image', img)

            # print("Timestamp: %d, hands: %d, fingers: %d" % (frame.timestamp, len(frame.hands), len(frame.fingers)))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                save_gesture = not save_gesture
                print("Saving frames %s.." % ("finished", "started")[save_gesture])


def main():
    controller = Leap.Controller()
    controller.set_policy_flags(Leap.Controller.POLICY_OPTIMIZE_HMD)
    controller.set_policy_flags(Leap.Controller.POLICY_IMAGES)
    try:
        run(controller)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == '__main__':
    main()
