import time
import pathlib
import os
import io
import numpy as np
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
import picamera
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from random import seed, random, randint

print(' > INIT')
# seed random number generator
seed(1)

# the TFLite converted to be used with edgetpu
model_file = 'model_edgetpu.tflite'

# The path to labels.txt that was downloaded with your model
label_file = 'labels.txt'


# Set up model
script_dir = pathlib.Path(__file__).parent.absolute()
label_path = os.path.join(script_dir, label_file)
model_path = os.path.join(script_dir, model_file)
# Number of times to run inference
inf_count = 3
# Mean value for input normalization
input_mean = 128.0
# STD value for input normalization
input_std = 128.0
# Max number of classification results
top_k = 1
# Classification score threshold
threshold = 0.1

def grab_image(img_size):

    # Create the in-memory image stream
    print(' > GRAB IMAGE', img_size)

    stream = io.BytesIO()
    with picamera.PiCamera() as camera:
        camera.resolution = (1024, 768)
        #camera.start_preview()
        time.sleep(2)
        camera.capture(stream, format='jpeg', resize=(img_size))
    # "Rewind" the stream to the beginning so we can read its content
    stream.seek(0)
    image = Image.open(stream)
    return image

def main():

    pygame.init()
    pygame.mouse.set_visible(False)

    display_width = 800
    display_height = 480

    gameDisplay = pygame.display.set_mode((display_width,display_height))
    #pygame.display.set_caption('A bit Racey')

    black = (0,0,0)
    white = (255,255,255)

    clock = pygame.time.Clock()

    def face(mood):
        if mood == 1:
            faceImg = pygame.image.load('face1.png')
        elif mood == 2:
            faceImg = pygame.image.load('face2.png')
        elif mood == 3:
            faceImg = pygame.image.load('face3.png')
        gameDisplay.blit(faceImg, (0, 0))

    def text_objects(text, font):
        textSurface = font.render(text, True, white)
        return textSurface, textSurface.get_rect()

    def message_display(text):
        largeText = pygame.font.Font('freesansbold.ttf', 40)
        TextSurf, TextRect = text_objects(text, largeText)
        #TextRect.center = ((display_width/2),(display_height/2))
        TextRect.center = ((display_width/2),(display_height - 100))
        gameDisplay.blit(TextSurf, TextRect)

    # while not crashed:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             crashed = True

    #     gameDisplay.fill(black)
    #     face(1)
    #     message_display('HELLO WORLD!')

    #     pygame.display.update()
    #     clock.tick(60)

    # pygame.quit()

    print(' > SETUP')
    print('Label path: ', label_path)
    print('Model path: ', model_path)


    print(' > READING LABELS')
    labels = read_label_file(label_path)

    print(' > MAKING INTERPRETER')
    interpreter = make_interpreter(model_path)
    print(' > ALLOCATING TENSORS')
    interpreter.allocate_tensors()

    # Model must be uint8 quantized
    if common.input_details(interpreter, 'dtype') != np.uint8:
        raise ValueError('Only support uint8 input type.')

    img_size = common.input_size(interpreter)

    while True:

        gameDisplay.fill(black)

        randface = randint(1, 2)
        face(randface)

        image = grab_image(img_size)

        # Image data must go through two transforms before running inference:
        # 1. normalization: f = (input - mean) / std
        # 2. quantization: q = f / scale + zero_point
        # The following code combines the two steps as such:
        # q = (input - mean) / (std * scale) + zero_point
        # However, if std * scale equals 1, and mean - zero_point equals 0, the input
        # does not need any preprocessing (but in practice, even if the results are
        # very close to 1 and 0, it is probably okay to skip preprocessing for better
        # efficiency; we use 1e-5 below instead of absolute zero).
        params = common.input_details(interpreter, 'quantization_parameters')
        scale = params['scales']
        zero_point = params['zero_points']
        mean = input_mean
        std = input_std
        if abs(scale * std - 1) < 1e-5 and abs(mean - zero_point) < 1e-5:
            # Input data does not require preprocessing.
            common.set_input(interpreter, image)
        else:
            # Input data requires preprocessing
            normalized_input = (np.asarray(image) - mean) / (std * scale) + zero_point
            np.clip(normalized_input, 0, 255, out=normalized_input)
            common.set_input(interpreter, normalized_input.astype(np.uint8))

        # Run inference
        # print('----INFERENCE TIME----')
        # print('Note: The first inference on Edge TPU is slow because it includes',
        #     'loading the model into Edge TPU memory.')
        for _ in range(inf_count):
            #start = time.perf_counter()
            interpreter.invoke()
            #inference_time = time.perf_counter() - start
            classes = classify.get_classes(interpreter, top_k, threshold)
            #print('%.1fms' % (inference_time * 1000))

        # print('-------RESULTS--------')
        for c in classes:
            print(labels.get(c.id, c.id))
            print (c.score)


        if c.score > 0.96:
            label = labels.get(c.id, c.id)
            if label == 'Achton':
                rand = randint(1, 4)
                if rand == 1:
                    message_display('No fear, the maker is here!')
                elif rand == 2:
                    message_display('Hvor er de 2 andre??')
                else:
                    message_display('Achton hacked me!')
            elif label == 'Øl':
                rand = randint(1, 4)
                if rand == 1:
                    message_display('BEER TIME')
                elif rand == 1:
                    message_display('Me likey!')
                else:
                    message_display('DRINK UP! SKÅÅÅÅÅL!')
            elif label == 'Alice':
                rand = randint(1, 4)
                if rand == 1:
                    message_display('ALICE min mesterbygger!')
                elif rand == 2:
                    message_display('Du har bygget mig!')
                else:
                    message_display("Hej %s!", label)
            elif label == 'Lisa':
                rand = randint(1, 4)
                if rand == 1:
                    message_display('LISA min mesterbygger!')
                elif rand == 2:
                    message_display('Du har bygget mig!')
                else:
                    message_display("Hej %s!", label)
            elif label == 'Fini':
                message_display("Hej %s!", label)
            else:
                message_display("Jeg synes jeg ser %s", label)

            #message_display(labels.get(c.id, c.id))
        else:
            message_display('')

        pygame.display.update()
        clock.tick(120)


if __name__ == '__main__':
  main()
