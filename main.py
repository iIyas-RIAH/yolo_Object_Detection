import cv2
import model_Yolo as yolo


model, classes, output_layers = yolo.load_yolo()
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('https://static.videezy.com/system/resources/previews/000/020/669/original/P1033658.mp4')


while True:
	_, frame = cap.read()
	height, width, channels = frame.shape
	blob, outputs = yolo.detect_objects(frame, model, output_layers)
	boxes, confs, class_ids = yolo.get_box_dimensions(outputs, height, width)
	yolo.draw_labels(boxes, confs, class_ids, classes, frame)
	key = cv2.waitKey(1)
	if key == 27:
		break
cap.release()