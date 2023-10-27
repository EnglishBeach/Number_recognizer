# %%
# %matplotlib qt
# %matplotlib ipympl

# %%
## Imports
print("Importing...")
import re
import os
import pandas as pd
from tqdm import tqdm

import cv2
import easyocr

from recognizer_modules import PreProcessor, PostProcessor, save_data

EXP_PATH, VIDEO_NAME, DATA_NAME = '', '', ''

# %%
## Inputs
EXP_PATH = r'Experiments\MultiplyTemperature\Exp1(2.5)'
VIDEO_NAME = r"\Exp1_up.avi"

variable_patterns = {
    'Viscosity': r'-?\d{1,3}\.\d',
    'Temperature': r'-?\d{1,3}\.\d',
}

# %%
if EXP_PATH + VIDEO_NAME == '':
    input_path = ''
    while (input_path == '') and (not os.path.isfile(EXP_PATH + VIDEO_NAME)):
        input_path = input(f"Input video path: ")
    path_list = (input_path).split('\\')
    EXP_PATH = '\\'.join(path_list[:-1])
    VIDEO_NAME = '\\' + path_list[-1]
DATA_NAME = VIDEO_NAME.split('.')[0] + '.csv'

print(
    'Recognize path:',
    EXP_PATH + VIDEO_NAME,
    f'Data save path:',
    EXP_PATH + DATA_NAME,
    sep='\n',
)


# %%
## PreProcessor settings
class ImageProcessor(PreProcessor):
    Blur = range(1, 50)

    def process(self, image, gray_image=True):
        image = cv2.blur(image, (int(self['Blur']), int(self['Blur'])))
        if gray_image:
            try:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            except:
                pass
            image = cv2.bitwise_not(image)

        return image


CAP = cv2.VideoCapture(EXP_PATH + VIDEO_NAME)

FPS = int(CAP.get(cv2.CAP_PROP_FPS))
LENTH = int(CAP.get(cv2.CAP_PROP_FRAME_COUNT) / FPS)
CAP.set(cv2.CAP_PROP_POS_FRAMES, 0)
_, START_FRAME = CAP.read()

processor = ImageProcessor([i for i in variable_patterns])
print('Configurate image processing')
processor.configure_process(CAP)
print(
    'Press:',
    '   Enter - save selection and continue',
    '   R     - reset video timer',
    '   Ecs/C - cancel selection',
    '   q/e   - time move',
    sep='\n',
)
processor.select_window(CAP)
processor.check_process(CAP)


# %%
## PostProcessor settings
class ValuePostProcessor(PostProcessor):

    def pattern_check(self, value: list, pattern: str):

        if value == []: return None
        value = value[0]
        value = value.replace(',', '.')
        if len(re.findall(pattern, value)) == 1:
            try:
                result = float(value)
                return result
            except ValueError:
                print('\nStrange error',re.findall(pattern, value)[0])
                return None

    @PostProcessor.check_type
    def processor_sweep(self) -> list[str]:
        for i in range(1, 50):
            self.inner_processor['Blur'] = i
            processed_img = self.inner_processor(self.image)
            raw_value = [
                value for _, value, _ in self.reader.readtext(processed_img)
            ]
            result = self.pattern_check(raw_value, self.pattern)
            if result is not None: return raw_value
        return []

    @PostProcessor.check_type
    def value_combine(self) -> list[str]:
        parts = len(self.raw_value)
        if parts == 1:
            value = self.raw_value[0]
            result = value[:3] + '.' + value[4:5]

        elif parts == 2:
            result = '.'.join(self.raw_value)

        elif parts == 3:
            result = f'{self.raw_value[0]}.{self.raw_value[2]}'

        return [result]


print('Starting recognizer...')
reader = easyocr.Reader(['en'])
checker = ValuePostProcessor(reader=reader, processor=processor)
# checker.active_checks_order = {check:checker.all_checks[check] for check in ['inner_processor_check','value_combine']}
print([i for i in checker.active_checks_order])
# %%
## Recognize
input_fps = input('Input number of frames per second: ')
try:
    read_fps = float(input_fps)
except:
    read_fps = 1

print('Recognizing:')
errors = 0
frame_line = tqdm(iterable=range(0, FPS * LENTH, int(FPS / read_fps)))
frame_line.set_description(f'Errors: {errors: >4}')
data = []

for i_frame in frame_line:
    CAP.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
    _, frame = CAP.read()
    i_text = {'time': round(i_frame / FPS, 1)}
    processed_frame = processor(frame)
    stricted_images = processor.strict(processed_frame)

    for var, pattern in variable_patterns.items():
        var_image = stricted_images[var]
        raw_value = [value for _, value, _ in reader.readtext(var_image)]

        mark, result = checker.check(image=var_image,
                               raw_value=raw_value,
                               pattern=pattern)
        # if mark == 'error':
        #     # processor.configure_process(CAP,start_frame=i_frame)
        #     processor.select_window(CAP,start_frame=i_frame)
        #     # processor.check_process(CAP,start_frame=i_frame)
        #     checker.reload_processor(processor)
        #     mark, result = checker.check(image=var_image,
        #                 raw_value=raw_value,
        #                 rules=rules)
        #     mark= f'*{mark}'

        i_text[var] = result
        i_text[var + '_verbose'] = mark

    if None in i_text.values():
        errors += 1
        frame_line.set_description(f'Errors: {errors: >4}')
    data.append(i_text)

# %%
## Saving
df = pd.DataFrame(data)
save_data(df, EXP_PATH + DATA_NAME)
