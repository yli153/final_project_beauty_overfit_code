import tensorflow as tf
import numpy as np
import image
from datetime import datetime
from tensorflow import keras
import color_transfer
                  
content_layers = ['block4_conv2']
style_layers = ['block1_conv1', 'block2_conv1','block3_conv1', 'block4_conv1','block5_conv1']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def model_init(algorithm):   
    vgg19 = tf.keras.applications.VGG19(include_top = False, weights = 'imagenet')
    vgg19.trainable = False
    content_output = []
    style_output = []

    
    if algorithm == 'portrait':
        global content_layers
        global num_content_layers
        content_layers = style_layers
        num_content_layers = num_style_layers

    for layer in content_layers:
        output = vgg19.get_layer(layer).output
        content_output.append(output)

    for layer in style_layers:
        output = vgg19.get_layer(layer).output
        style_output.append(output)

    output = content_output + style_output
    model = keras.models.Model(vgg19.input, output)
    
    return model    


def content_loss(algorithm, base_content, target):
    content_loss = tf.reduce_mean(tf.square(base_content - target))/2

    if algorithm == 'portait':
        base_content = tf.convert_to_tensor(base_content)
        h,w,c = base_content.get_shape().as_list()
        content_loss = content_loss/(h*w*c)

    return content_loss

def gram_matrix(input_tensor):
    channel = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor,[-1,channel]) 
    gram_matrix = tf.matmul(tf.transpose(a), a)
    return gram_matrix

def style_loss(algorithm, base_style, target):
    a = gram_matrix(base_style)
    b = gram_matrix(target)

    if algorithm == 'original_nst':
        h,w,c = base_style.get_shape().as_list()
        style_loss = (1. /(4*(h**2)*(w**2)*(c**2))) * tf.reduce_mean(tf.pow((a - b),2))
    
    if algorithm == 'portrait':
        c = base_style.get_shape().as_list()[2]
        style_loss = (1./(2*(c**2))) * tf.reduce_mean(tf.pow((a - b),2))

    return style_loss


def get_feature(model, style_path, content_path, algorithm, mode, color):
    content = image.load_and_preprocess(content_path)

    if mode == 'original' or mode == 'color_trans':
        style = image.load_and_preprocess(style_path)
        if color == 'cholesky' or color == 'image_analogies':
            # creating new style matrix for color transfer
            style = np.squeeze(style, axis=0)
            content = np.squeeze(content, axis=0)
            new_style = color_transfer.pixel_transformation(color, style, content)
            new_style = np.expand_dims(new_style, axis=0)
            content = np.expand_dims(content, axis=0)
            style_feature_outputs = model(new_style)
            content_feature_outputs = model(content)
        else:
            style_feature_outputs = model(style)
            content_feature_outputs = model(content)

        style_feature_arr, content_feature_arr = [], []

        for feature in style_feature_outputs[num_content_layers:]:
            style_feature_arr.append(feature[0])
        
        for feature in content_feature_outputs[:num_content_layers]:
            content_feature_arr.append(feature[0])

    elif mode == 'artist_style':
        styles = []
        for path in style_path:
            style = image.load_and_preprocess(path)
            style = np.squeeze(style, axis=0)
            styles.append(style)

        content = np.squeeze(content, axis=0)
        new_styles = []
        for style in styles:
            new_style = color_transfer.pixel_transformation(color, style, content)
            new_style = np.expand_dims(new_style, axis=0)
            new_styles.append(new_style)
        content = np.expand_dims(content, axis=0)
        artist_style_feature_outputs = []

        for new_style in new_styles:
            style_feature_outputs = model(new_style)
            artist_style_feature_outputs.append(style_feature_outputs)

        content_feature_outputs = model(content)
    
        style_feature_arr, content_feature_arr = [], []

        for image_style_feature_outputs in artist_style_feature_outputs:
            style_feature = []
            for feature in image_style_feature_outputs[num_content_layers:]:
                style_feature.append(feature[0])
            style_feature_arr.append(style_feature)
    
        for feature in content_feature_outputs[:num_content_layers]:
            content_feature_arr.append(feature[0])
    
    if algorithm == 'portrait':
        if mode == 'original' or mode == 'color_trans':
            new_content_feature_arr = []

            for i in range(num_content_layers):
                gain_map = style_feature_arr[i]/np.add(content_feature_arr[i] , 10**(-4))
                gain_map = np.clip(gain_map, 0.7, 5)
                new_content_feature_arr.append(np.multiply(content_feature_arr[i], gain_map))

            content_feature_arr = new_content_feature_arr
        
        artist_content_feature_arr = []
        
        if mode == 'artist_style':
            for style_feature in style_feature_arr:
                new_content_feature_arr = []
            for i in range(num_content_layers):
                gain_map = style_feature[i]/np.add(content_feature_arr[i] , 10**(-4))
                gain_map = np.clip(gain_map, 0.7, 5)
                new_content_feature_arr.append(np.multiply(content_feature_arr[i], gain_map))
            artist_content_feature_arr.append(new_content_feature_arr)
            content_feature_arr = artist_content_feature_arr
    
    return (style_feature_arr, content_feature_arr)
    

def loss(model, content_features, style_features, loss_weights, gen_image, algorithm, mode):

    total_style_loss = 0
    total_content_loss = 0

    features = model(gen_image)
    gen_style_feature = features[num_content_layers:]
    gen_content_feature = features[:num_content_layers]
    
    style_weight,content_weight = loss_weights

    if algorithm == 'original_nst':
        if mode == 'original' or mode == 'color_trans':
            style_layer_weight = 1.0 / float(num_style_layers)
            for i in range(len(style_features)): 
                total_style_loss = total_style_loss + style_layer_weight * style_loss(algorithm, style_features[i], gen_style_feature[i])
        
        elif mode == 'artist_style':
            style_layer_weight = 1.0 / float(num_style_layers)
            style_painting_weight = 1.0 / float(len(style_features))
            for style in style_features:
                tmp_style_loss = 0
                for i in range(len(style)): 
                    tmp_style_loss = tmp_style_loss + style_layer_weight * style_loss(algorithm, style[i], gen_style_feature[i])
                total_style_loss += style_painting_weight * tmp_style_loss

        content_layer_weight = 1.0 / float(num_content_layers)
        for i in range(len(content_features)): 
            total_content_loss = total_content_loss + content_layer_weight * content_loss(algorithm, content_features[i], gen_content_feature[i])
    
    if algorithm == 'portrait':
        if mode == 'original' or mode == 'color_trans':
            style_layer_weight = 1.0
            for i in range(len(style_features)): 
                if i == 2 or i == 3:
                    style_layer_weight = 0.5
                total_style_loss = total_style_loss + style_layer_weight * style_loss(algorithm, style_features[i], gen_style_feature[i])
            

            content_layer_weight = 1.0
            for i in range(len(content_features)): 
                if i == 2 or i == 3:
                    content_layer_weight = 0.5
                total_content_loss = total_content_loss + content_layer_weight * content_loss(algorithm, content_features[i], gen_content_feature[i])

        if mode == 'artist_style':
            init_style_layer_weight = 1.0 / float(num_style_layers)
            style_layer_weight = init_style_layer_weight
            style_painting_weight = 1.0 / float(len(style_features))
            
            for style in style_features:
                tmp_style_loss = 0
                for i in range(len(style)):
                    if i == 2 or i == 3:
                        style_layer_weight = 0.5
                    else:
                        style_layer_weight = init_style_layer_weight
                    tmp_style_loss = tmp_style_loss + style_layer_weight * style_loss(algorithm, style[i], gen_style_feature[i])
                total_style_loss += style_painting_weight * tmp_style_loss
            

            init_content_layer_weight = 1.0 / float(num_content_layers)
            content_layer_weight = init_content_layer_weight
            content_painting_weight = 1.0 / float(len(content_features))

            for content in content_features:
                tmp_content_loss = 0
                for i in range(len(content)):
                    if i == 2 or i == 3:
                        content_layer_weight = 0.5
                    else:
                        content_layer_weight = init_content_layer_weight
                    tmp_content_loss = tmp_content_loss + content_layer_weight * content_loss(algorithm, content[i], gen_content_feature[i])
                total_content_loss += content_painting_weight * tmp_content_loss

    total_style_loss = total_style_loss * style_weight
    total_content_loss = total_content_loss * content_weight
    total_loss = total_content_loss + total_style_loss
    return total_loss, total_content_loss, total_style_loss
    

def compute_gradients(cfg):
    with tf.GradientTape() as tape:
        all_loss = loss(**cfg)
    #Compute Gradient with respect to the generated image
    total_loss = all_loss[0]
    return tape.gradient(total_loss,cfg['gen_image']),all_loss


def run(content_path, style_path, algorithm, mode, color, artist, iteration):

    gen_image = image.load_and_preprocess(content_path) 
    gen_image = tf.Variable(gen_image, dtype = tf.float32)

    content_weight = 1e3
    style_weight = 1

    model = model_init(algorithm)

    for layer in model.layers:
        layer.trainable = False
    
    style_features, content_features = get_feature(model, style_path, content_path, algorithm, mode, color)
    
    #parameter suggested by tensorflow 
    optimizer = tf.keras.optimizers.Adam(2, beta_1 = 0.99, epsilon = 1e-1)
    
    loss_weights = (content_weight,style_weight)
    
    cfg = { 'model':model, 'content_features':content_features, 'style_features':style_features, 'loss_weights':loss_weights,  'gen_image':gen_image, 'algorithm': algorithm, 'mode': mode }
    
    #pixl means given by imagenet
    pm = [103.939, 116.779, 123.68]
    pixel_means = np.asarray(pm)
    maxvalues = 255 - pixel_means
    minvalues = -pixel_means
    
    best_loss = float('inf')

    for i in range(iteration):

        grads, all_loss = compute_gradients(cfg)
        total_losss, content_losss, style_losss = all_loss
        optimizer.apply_gradients([(grads, gen_image)])
        clipped = tf.clip_by_value(gen_image, minvalues, maxvalues)
        gen_image.assign(clipped)
        
        if total_losss < best_loss:
            best_loss = total_losss
            best_img = image.deprocess(gen_image.numpy())

        if i % 100 == 0:
            print('Iteration: {}'.format(i))
            print('Total loss: {:.4e}, ' 'content loss: {:.4e}, '  'style loss: {:.4e}'.format(total_losss, content_losss, style_losss))

            path = 'output/output_' + str(i) +'_'+ algorithm+'_'+mode+'_'+color+'_'+artist+'.jpg'
            img = gen_image.numpy()
            img = image.deprocess(img)
            image.save(img, path)


    return best_img, best_loss
    
