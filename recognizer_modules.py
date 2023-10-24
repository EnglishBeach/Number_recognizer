import copy
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class PreProcessor:
    _parametr_configurations = []
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
        for parametr, diap in self._parametr_configurations.items():
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

        print('Configurate image processing')
        plt.show()

    def select_window(self, video_capture, i_frame=0):
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
        _, image = video_capture.read()

        for variable in self.variable_windows:
            processed_image = self.process(image)
            processed_image = cv2.bitwise_not(processed_image)
            # TODO: make border color=red
            window = cv2.selectROI(
                f"Select {variable}",
                processed_image,
                fromCenter=False,
                showCrosshair=True,
            )
            self.variable_windows[variable] = window
        cv2.destroyAllWindows()

    def check_process(self,
                      video_capture,
                      start_frame: int = 0,
                      end_frame: int = 0):

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
            x, y, dx, dy = window
            images[variable] = image[y:y + dy, x:x + dx]
        return images

    def process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __init__(self, variables):
        all_fields = dict(self.__class__.__dict__)
        self._parametr_configurations = {
            key: value
            for key, value in all_fields.items()
            if key[0].isupper()
        }
        self.parametrs = {
            key: min(value)
            for key,
            value in self._parametr_configurations.items()
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

    def check(self, image, raw_value, rules):
        pattern_check = self.pattern(raw_value, **rules)
        if pattern_check is not None: return '', pattern_check

        self._rules = rules
        self._image = image
        self._raw_value = raw_value
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

    @_check_type
    def inner_processor_check(self) -> list[str]:
        processed_image = self.inner_processor(self._image)

        raw_value = [
            value for _, value, _ in self._reader.readtext(processed_image)
        ]
        return raw_value