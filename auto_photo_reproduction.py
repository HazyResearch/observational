# -*- coding: utf-8 -*-
import sys
import os
import pygame
from pygame.locals import *
import datetime
from PIL import Image
import csv


def main(screen):
    pygame.display.set_caption("犬を見る")  
    screen.fill((230,230,230))
    pygame.display.update()
    time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    csv_file = open(time + '.csv', 'w')
    writer = csv.writer(csv_file)
    filenames = os.listdir('fig')
    os.chdir('fig')
    width, height = screen.get_size()
    
    isEnd = True
    while isEnd:        
        for filename in filenames:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN:  
                
                    pygame.quit()
                    sys.exit()
            img = Image.open(filename)
            x, y = img.size
            mag = float(height) / y
            img = pygame.image.load(filename)
            img = pygame.transform.scale(img, (int(x * mag), int(y * mag)))
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            writer.writerow([time, filename])
            screen.blit(img, ((width - int(x * mag))/ 2 , (height - int(y * mag))/ 2))
            pygame.display.update()
            pygame.time.wait(6000)
        else:
            isEnd = False

if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode()
    main(screen)