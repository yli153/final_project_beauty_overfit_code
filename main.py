import tensorflow as tf
import os
import image
import model
import argparse

content_path = 'input/content/james.jpg'

artist1_style_path = 'input/style/artist1/Vincent_van_Gogh_604.jpg'
artist1_style_path_arr = [
    'input/style/artist1/Vincent_van_Gogh_52.jpg',
    'input/style/artist1/Vincent_van_Gogh_69.jpg', 
    'input/style/artist1/Vincent_van_Gogh_210.jpg',
    'input/style/artist1/Vincent_van_Gogh_219.jpg',
    'input/style/artist1/Vincent_van_Gogh_604.jpg']

artist2_style_path = 'input/style/artist2/Leonardo_da_Vinci_121.jpg'
artist2_style_path_arr = [
    'input/style/artist2/Leonardo_da_Vinci_34.jpg',
    'input/style/artist2/Leonardo_da_Vinci_52.jpg', 
    'input/style/artist2/Leonardo_da_Vinci_93.jpg',
    'input/style/artist2/Leonardo_da_Vinci_121.jpg',
    'input/style/artist2/Leonardo_da_Vinci_129.jpg']

artist3_style_path = 'input/style/artist3/Amedeo_Modigliani_6.jpg'
artist3_style_path_arr = [
    'input/style/artist3/Amedeo_Modigliani_24.jpg',
    'input/style/artist3/Amedeo_Modigliani_29.jpg', 
    'input/style/artist3/Amedeo_Modigliani_35.jpg',
    'input/style/artist3/Amedeo_Modigliani_6.jpg',
    'input/style/artist3/Amedeo_Modigliani_74.jpg']


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-al', '--algorithm', default='original_nst', help='Either original_nst, or portrait')
    parser.add_argument('-m', '--mode', default='original', help='Either original, color_trans, or artist_style')
    parser.add_argument('-c', '--color', default='original', help='Either original, cholesky, or image_analogies')
    parser.add_argument('-a', '--artist', default='artist1', help='Either artist1, artist2, or artist3')

    args = parser.parse_args()
    
    if args.mode =='original' and args.color!='original':
        raise AssertionError()
    
    if args.mode == 'color_trans' or args.mode == 'artist_style':
        if args.color == 'original':
            raise AssertionError()

    if args.mode != 'artist_style':

        if args.artist == 'artist1':
            style_path = artist1_style_path
        elif args.artist == 'artist2':
            style_path = artist2_style_path
        elif args.artist == 'artist3':
            style_path = artist3_style_path

        best, best_loss = model.run(content_path, style_path, args.algorithm, args.mode, args.color, args.artist, iteration=1000)
    else:

        if args.artist == 'artist1':
            style_path_arr = artist1_style_path_arr
        elif args.artist == 'artist2':
            style_path_arr = artist2_style_path_arr
        elif args.artist == 'artist3':
            style_path_arr = artist3_style_path_arr

        best, best_loss = model.run(content_path, style_path_arr, args.algorithm, args.mode, args.color, args.artist, iteration=1000)
    image.save(best, 'output/best'+'_'+args.algorithm+'_'+args.mode+'_'+args.color+'_'+args.artist+ '.jpg')
