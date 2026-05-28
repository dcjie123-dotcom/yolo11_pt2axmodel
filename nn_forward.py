from maix import camera, display, image, nn, app, time

detector = nn.YOLO11(model="/root/models/train15_2026.5.21/yolo11_detect.mud", dual_buff = True)

cam = camera.Camera(detector.input_width(), detector.input_height(), detector.input_format(),fps=-1,buff_num=1)
disp = display.Display()

while not app.need_exit():
    print("66")
    img = cam.read()

    objs = detector.detect(img, conf_th = 0.5, iou_th = 0.45)
    print("23")
    for obj in objs:
        img.draw_rect(obj.x, obj.y, obj.w, obj.h, color = image.COLOR_RED)
        msg = f'{detector.labels[obj.class_id]}: {obj.score:.2f}'
        img.draw_string(obj.x, obj.y, msg, color = image.COLOR_RED,scale=3,thickness=3)
    fps = time.fps()
    img.draw_string(10, 10, f"fps: {fps:.02f}", color = image.COLOR_RED,scale=3,thickness=3)
    disp.show(img)
    print("44")

print("45")
