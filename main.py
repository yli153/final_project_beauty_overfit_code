#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import os
import image
import model
import argparse

content_path = 'input/content/james.jpg'
style_path = 'input/style/Vincent_van_Gogh_604.jpg'
style_path_arr = [
    'input/style/Vincent_van_Gogh_59.jpg',
    'input/style/Vincent_van_Gogh_69.jpg', 
    'input/style/Vincent_van_Gogh_210.jpg',
    'input/style/Vincent_van_Gogh_219.jpg',
    'input/style/Vincent_van_Gogh_604.jpg']

#image.py用的github上的原码，我过几天再更新一版

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algorithm', default='original_nst', help='Either original_nst, or portrait')
    parser.add_argument('-m', '--mode', default='original', help='Either original, color_trans, or artist')
    parser.add_argument('-c', '--color', default='original', help='Either original, cholesky, or image_analogies')

    args = parser.parse_args()
    
    if args.mode =='original' and args.color!='original':
        raise AssertionError()
    
    if args.mode == 'color_trans' or args.mode == 'artist':
        if args.color == 'original':
            raise AssertionError()

    if args.mode != 'artist':
        best, best_loss = model.run(content_path, style_path, args.algorithm, args.mode, args.color, iteration=1000)
    else:
        best, best_loss = model.run(content_path, style_path_arr, args.algorithm, args.mode, args.color, iteration=1000)
    image.saveimg(best, 'output/best.jpg')
