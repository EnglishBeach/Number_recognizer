import copy
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class PreProcessor:
    parametr_configurations = []
    parametrs = {}

    def configure_process(
        self,
        video_capture,
        start_frame: int = 0,
        end_frame: int = 0,
    ):

        def update(val):
            time = TIME_slider.val
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, int(fps * time))
            _, image = video_capture.read()

            for slider in sliders:
                slider_name = str(slider.label).split("'")[1]
                self[slider_name] = slider.val

            image_processed = self.process(image)
            plot.set_data(image_processed)
            plot.autoscale()
            fig.canvas.draw_idle()

        fig, ax = plt.subplots()
        fig.set_size_inches(5, 5)
        fig.subplots_adjust(
            left=0.25,
            right=1,
            bottom=0.25,
            top=1,
            hspace=0,
            wspace=0,
        )
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _, image = video_capture.read()

        plot = ax.imshow(self.process(image), cmap='binary')

        time_slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        max_len = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps) - 1
        end_frame = max_len if not end_frame else end_frame

        TIME_slider = Slider(
            ax=time_slider_ax,
            label='Time',
            valmin=start_frame,
            valmax=end_frame,
            valinit=start_frame,
            valstep=1,
        )
        TIME_slider.on_changed(update)

        sliders = []
        ofset = 0.2
        for parametr, diap in self.parametr_configurations.items():
            slider_ax = fig.add_axes([ofset, 0.25, 0.03, 0.6])

            p_min = min(diap)
            p_max = max(diap)
            p_step = (max(diap) - min(diap)) / (len(diap) - 1)

            slider = Slider(
                ax=slider_ax,
                orientation='vertical',
                label=parametr,
                valmin=p_min,
                valmax=p_max,
                valinit=self[parametr],
                valstep=p_step,
            )
            slider.on_changed(update)
            sliders.append(slider)
            ofset -= 0.02


        plt.show()

    def _build_selection_window(
        self,
        video_capture,
        window_name: str = 'Selection window',
        start_frame: int = 0,
    ):

        cv2.namedWindow(window_name)
        # if not cap.isOpened():
        #     raise NameError

        # Our ROI, defined by two points
        point0, point1 = (0, 0), (0, 0)
        drawing = False  # True while ROI is actively being drawn by mouse
        show_drawing = False  # True while ROI is drawn but is pending use or cancel
        blue_color = (255, 0, 0)

        def on_mouse(event, x, y, flags, userdata):
            nonlocal point0, point1, drawing, show_drawing
            if event == cv2.EVENT_LBUTTONDOWN:
                # Left click down (select first point)
                drawing = True
                show_drawing = True
                point0 = x, y
                point1 = x, y
            elif event == cv2.EVENT_MOUSEMOVE:
                # Drag to second point
                if drawing:
                    point1 = x, y
            elif event == cv2.EVENT_LBUTTONUP:
                # Left click up (select second point)
                drawing = False
                point1 = x, y

        cv2.setMouseCallback(window_name, on_mouse)

        while True:
            capture_ready, frame = video_capture.read()
            # Reset timer when video ends
            if not capture_ready:
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                capture_ready, frame = video_capture.read()
            frame = self.process(frame, gray_image=False)

            # Show rectangle
            if show_drawing:
                point1 = (0 if point1[0] < 0 else
                          (point1[0]
                           if point1[0] < frame.shape[1] else frame.shape[1]),
                          0 if point1[1] < 0 else
                          (point1[1]
                           if point1[1] < frame.shape[0] else frame.shape[0]))

                cv2.rectangle(frame, point0, point1, blue_color, 2)
            cv2.imshow(window_name, frame)

            keyboard = cv2.waitKey(1)
            # Pressed Enter or Space to cunsume
            if keyboard in [13, 32]:
                drawing = False
                cv2.destroyAllWindows()
                break

            # Pressed C or Esc to cancel selection
            elif keyboard in [ord('c'), ord('C'), 27]:
                point0 = (0, 0)
                point1 = (0, 0)

            # Pressed r to reset video timer
            elif keyboard in [ord('r'), ord('R')]:
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        cv2.destroyAllWindows()
        return point0, point1

    def select_window(self, video_capture, start_frame: int = 0):
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        _, image = video_capture.read()
        for variable in self.variable_windows:
            point0,point1 = self._build_selection_window(
                video_capture,window_name=f"Select {variable}",
                start_frame=start_frame,
                )
            if (point0, point1) == ((0, 0), (0, 0)):
                point1 = image.shape[:2:][::-1]
            self.variable_windows[variable] = (point0, point1)

    def check_process(
        self,
        video_capture,
        start_frame: int = 0,
        end_frame: int = 0,
    ):

        def update(val):
            time = TIME_slider.val
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, int(fps * time))
            _, image = video_capture.read()

            image_processed = self.process(image)
            stricted_images_list = self.strict(image_processed)

            i = 0
            for variable in self.variable_windows:
                plots[i].set_data(stricted_images_list[variable])
                plots[i].autoscale()
                i += 1

            fig.canvas.draw_idle()

        fig, axises = plt.subplots(nrows=len(self.variable_windows) + 1)
        fig.set_size_inches(5, 5)
        fig.subplots_adjust(
            left=0.1,
            right=0.9,
            bottom=0.0,
            top=1,
            hspace=0.0,
            wspace=0.1,
        )
        if not isinstance(axises, np.ndarray): axises = [axises]

        time_slider_ax = axises[0]
        axises = axises[1:]

        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _, image = video_capture.read()
        image_processed = self.process(image)
        stricted_images_list = self.strict(image_processed)

        plots = []
        i = 0
        for variable in self.variable_windows:
            plots.append(
                axises[i].imshow(
                    stricted_images_list[variable],
                    cmap='binary',
                ), )
            i += 1
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        max_len = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps) - 1
        end_frame = max_len if not end_frame else end_frame

        TIME_slider = Slider(
            ax=time_slider_ax,
            label='Time',
            valmin=start_frame,
            valmax=end_frame,
            valinit=start_frame,
            valstep=1,
        )
        TIME_slider.on_changed(update)
        plt.show()

    def strict(self, image: np.ndarray) -> np.ndarray:
        images = {}
        for variable, window in self.variable_windows.items():
            (x0, y0), (x1, y1) = window
            X, Y = (x0, x1), (y0, y1)
            x0, x1 = min(X), max(X)
            y0, y1 = min(Y), max(Y)

            images[variable] = image[y0:y1, x0:x1]
        return images

    def process(self, image: np.ndarray, gray_image=True) -> np.ndarray:
        raise NotImplementedError

    def __init__(self, variables):
        all_fields = dict(self.__class__.__dict__)
        self.parametr_configurations = {
            key: value
            for key, value in all_fields.items()
            if key[0].isupper()
        }
        self.parametrs = {
            key: min(value)
            for key,
            value in self.parametr_configurations.items()
        }
        self.variable_windows = {variable: 0 for variable in variables}

    def __getitem__(self, item):
        return self.parametrs[item]

    def __setitem__(self, item, value):
        self.parametrs[item] = value

    def __call__(self, image) -> np.ndarray:
        return self.process(image)


class PostProcessor:
    _image = None
    _raw_value = []
    _rules = dict(re_rule=None, min_rule=None, max_rule=None)
    active_checks_order = {}

    def check(self, image, raw_value, rules, inside_parametrs={}):
        pattern_check = self.pattern(raw_value, **rules)
        if pattern_check is not None: return 'OK', pattern_check

        self._raw_value = raw_value
        self._rules = rules
        self._image = image

        self._inside_parametrs = inside_parametrs

        for check_name, check_func in self.active_checks_order.items():
            check_result = check_func(self)
            result = self.pattern(check_result, **rules)
            if result is not None: return check_name, result
        return 'error', None

    @staticmethod
    def _check_type(func=None, get=False, checks={}):
        if func is not None: checks.update({func.__name__: func})
        if get: return checks
        return func

    @_check_type
    def inner_processor_check(self) -> list[str]:
        processed_image = self.inner_processor(self._image)

        raw_value = [
            value for _, value, _ in self._reader.readtext(processed_image)
        ]
        return raw_value

    def __init__(
        self,
        processor: PreProcessor,
        reader,
    ):
        self.inner_processor = copy.deepcopy(processor)
        self._reader = reader
        self.active_checks_order = self._check_type(get=True)

    def pattern(self, value: list) -> float|None:
        raise NotImplementedError

    @property
    def all_checks(self):
        return self._check_type(get=True)

    def reload_processor(self, processor):
        self.inner_processor = copy.deepcopy(processor)
