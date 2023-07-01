import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils import get_max_lengths, get_evaluation
from src.dataset import MyDataset
from src.hierarchical_att_model import HierAttNet
import argparse
import shutil
import numpy as np
import csv
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


def visualize_attention(doc, scores, alphas, sentence_alphas, label_map):
   
    """
    https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Text-Classification/blob/master/classify.py
    Visualize important sentences and words, as seen by the HAN model.

    :param doc: pre-processed tokenized document
    :param scores: class scores, a tensor of size (n_classes)
    :param word_alphas: attention weights of words, a list of arbitrary lengths (n_sentences, arbitraty lengths)
    :param sentence_alphas: attention weights of sentences, a tensor of size (n_sentences)
    :param words_in_each_sentence: sentence lengths, a tensor of size (n_sentences)
    """
    # Find best prediction
    score, prediction = scores.max(dim=0)
    prediction = '{category} ({score:.2f}%)'.format(category=label_map[prediction.item()], score=score.item() * 100)

    # For each word, find it's effective importance (sentence alpha * word alpha)
    # alphas = (sentence_alphas.unsqueeze(1) * word_alphas * words_in_each_sentence.unsqueeze(
    #    1).float() / words_in_each_sentence.max().float())
    # alphas = word_alphas * words_in_each_sentence.unsqueeze(1).float() / words_in_each_sentence.max().float()
    #alphas = alphas.to('cpu')

    # Determine size of the image, visualization properties for each word, and each sentence
    min_font_size = 15  # minimum size possible for a word, because size is scaled by normalized word*sentence alphas
    max_font_size = 55  # maximum size possible for a word, because size is scaled by normalized word*sentence alphas
    space_size = ImageFont.truetype("./calibril.ttf", max_font_size).getsize(' ')  # use spaces of maximum font size
    line_spacing = 15  # spacing between sentences
    left_buffer = 100  # initial empty space on the left where sentence-rectangles will be drawn
    top_buffer = 2 * min_font_size + 3 * line_spacing  # initial empty space on the top where the detected category will be displayed
    image_width = left_buffer  # width of the entire image so far
    image_height = top_buffer + line_spacing  # height of the entire image so far
    word_loc = [image_width, image_height]  # top-left coordinates of the next word that will be printed
    rectangle_height = 0.75 * max_font_size  # height of the rectangles that will represent sentence alphas
    max_rectangle_width = 0.8 * left_buffer  # maximum width of the rectangles that will represent sentence alphas, scaled by sentence alpha
    rectangle_loc = [0.9 * left_buffer,
                     image_height + rectangle_height]  # bottom-right coordinates of next rectangle that will be printed
    word_viz_properties = list()
    sentence_viz_properties = list()
    for s, sentence in enumerate(doc):
        # Find visualization properties for each sentence, represented by rectangles
        # Factor to scale by
        sentence_factor = sentence_alphas[s].item() / sentence_alphas.max().item()

        # Color of rectangle
        rectangle_saturation = str(int(sentence_factor * 100))
        rectangle_lightness = str(25 + 50 - int(sentence_factor * 50))
        rectangle_color = 'hsl(0,' + rectangle_saturation + '%,' + rectangle_lightness + '%)'

        # Bounds of rectangle
        rectangle_bounds = [rectangle_loc[0] - sentence_factor * max_rectangle_width,
                            rectangle_loc[1] - rectangle_height] + rectangle_loc

        # Save sentence's rectangle's properties
        sentence_viz_properties.append({'bounds': rectangle_bounds.copy(),
                                        'color': rectangle_color})

        for w, word in enumerate(sentence):
            # Find visualization properties for each word
            # Factor to scale by
            word_factor = alphas[s][0, w].item() / alphas[s].view(-1).max().item()

            # Color of word
            word_saturation = str(int(word_factor * 100))
            word_lightness = str(25 + 50 - int(word_factor * 50))
            word_color = 'hsl(0,' + word_saturation + '%,' + word_lightness + '%)'

            # Size of word
            word_font_size = int(min_font_size + word_factor * (max_font_size - min_font_size))
            word_font = ImageFont.truetype("./calibril.ttf", word_font_size)

            # Save word's properties
            word_viz_properties.append({'loc': word_loc.copy(),
                                        'word': word,
                                        'font': word_font,
                                        'color': word_color})

            # Update word and sentence locations for next word, height, width values
            word_size = word_font.getsize(word)
            word_loc[0] += word_size[0] + space_size[0]
            image_width = max(image_width, word_loc[0])
        word_loc[0] = left_buffer
        word_loc[1] += max_font_size + line_spacing
        image_height = max(image_height, word_loc[1])
        rectangle_loc[1] += max_font_size + line_spacing

    # Create blank image
    img = Image.new('RGB', (image_width, image_height), (255, 255, 255))

    # Draw
    draw = ImageDraw.Draw(img)
    # Words
    for viz in word_viz_properties:
        draw.text(xy=viz['loc'], text=viz['word'], fill=viz['color'], font=viz['font'])
    # Rectangles that represent sentences
    for viz in sentence_viz_properties:
        draw.rectangle(xy=viz['bounds'], fill=viz['color'])
    # Detected category/topic
    category_font = ImageFont.truetype("./calibril.ttf", min_font_size)
    draw.text(xy=[line_spacing, line_spacing], text='Detected Category:', fill='grey', font=category_font)
    draw.text(xy=[line_spacing, line_spacing + category_font.getsize('Detected Category:')[1] + line_spacing],
              text=prediction.upper(), fill='black',
              font=category_font)
    del draw

    # Display
    img.save("attention_weight_images/image_4.jpg")
    #plt.imshow(img)
    #plt.savefig("attention_weight_images/image_1.jpg")
    #img.show()


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--pre_trained_model", type=str, default="trained_models/dbpedia/whole_model_han")
    parser.add_argument("--word2vec_path", type=str, default="data/glove.6B.50d.txt")
    parser.add_argument("--document",type = str)
    parser.add_argument("--gpu_id", type=int, default=0)    
    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    return args

def predict(opt,label_map):

    document = opt.document
    dict = pd.read_csv(filepath_or_buffer=opt.word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE,
                                usecols=[0]).values
    dict = [word[0] for word in dict]

    document_text =[ [word for word in word_tokenize(text=sentences)] 
                     for sentences in sent_tokenize(text=document)]
    document_encode = [ [dict.index(word)+1 if word in dict else 0 for word in word_tokenize(text=sentences)]
                         for sentences in sent_tokenize(text=document)]
    # document encode has arbitrary number of sentences and each sentence has arbitrary number of words

    if torch.cuda.is_available():
        model = torch.load(opt.pre_trained_model)
    else:
        model = torch.load(opt.pre_trained_model, map_location=lambda storage, loc: storage)

    # At this point all model learnable weights(weights of nn.Linear,nn.Embedding,nn.GRU,nn.LSTM,nn.conv2d ...) and other parameters(word_hidden_state,sent_hidden_state) are stored in device
    # which was used during training(in this case gpu:3)
    model = model.to(opt.device)

    # Now when we move model into opt.device, then only learnable weights will get moved, other parameters still remain in gpu:3
    print(model.word_hidden_state.device)
    print(model.sent_hidden_state.device)

    #document_encode = document_encode.to(opt.device)

    model._init_hidden_state(last_batch_size = 1)
    model.word_hidden_state = model.word_hidden_state.to(opt.device) # moving initial hidden states to 'device'
    model.sent_hidden_state = model.sent_hidden_state.to(opt.device)

    word_alpha = []
    output_list = []
    for i,sentence in enumerate(document_encode):
        # torch.tensor(sentence) = (k)
        output, model.word_hidden_state,alpha = model.word_att_net(torch.tensor(sentence).reshape(-1,1).to(opt.device), model.word_hidden_state) 
        #output_shape = (1,1, 2*hidden_size)
        # alpha = (1,k)     
        word_alpha.append(alpha)
        output_list.append(output)

        print("\nAttention weights of words in sentence_",i+1)
        for j,word in enumerate(document_text[i]):
            print(word,"{:.3f}".format(alpha[0,j].item()))
    
    output = torch.cat(output_list, 0) # shape = (#sentences, 1 , 2*hidden_size)
    output, model.sent_hidden_state, sent_alpha = model.sent_att_net(output, model.sent_hidden_state)
    output = nn.functional.softmax(output,dim = 1)
    # output = (1, num_classes)
    # sent_alpha = (1, #sentences)

    print("\n ----- Predicted_class : ------ {}\n".format(torch.argmax(output).item()))
    print("\nAttention weights for sentences \n")
    for i,sentence in enumerate(document_text):
        sentence = ' '.join(sentence)
        print(sentence,"{:.3f}".format(sent_alpha[0,i].item()))


    visualize_attention(document_text,output.view(-1),word_alpha,sent_alpha.view(-1),label_map)



if __name__ == "__main__":

    yahoo_label_map = ["Society & Culture",
                    "Science & Mathematics",
                    "Health",
                    "Education & Reference",
                    "Computers & Internet",
                    "Sports",
                    "Business & Finance",
                    "Entertainment & Music",
                    "Family & Relationships",
                    "Politics & Government"]

    dbpedia_label_map = ["Company",
                        "EducationalInstitution",
                        "Artist",
                        "Athlete",
                        "OfficeHolder",
                        "MeanOfTransportation",
                        "Building",
                        "NaturalPlace",
                        "Village",
                        "Animal",
                        "Plant",
                        "Album",
                        "Film",
                        "WrittenWork"]

    opt = get_args()

    # dbpedia, 
    # true class = 13
    #opt.document = " the lacuna is a 2009 novel by barbara kingsolver. it is kingsolver's sixth novel and won the 2010 orange prize for fiction and the library of virginia literary award. it was shortlisted for the 2011 international impac dublin literary award. "
    # true class 12
    #opt.document = " dorian gray is a 1970 movie adaptation of oscar wilde's novel the picture of dorian gray starring helmut berger. directed by massimo dallamano the film stresses the decadence and eroticism of the story and changes the setting to early 1970s london. "
    
    # yahoo answers, true class 8
    #opt.document = "how does the spark keep going? good communication is what does it.  can you move beyond small talk and say what's really on your mind.  if you start doing this, my expereince is that potentially good friends will respond or shun you.  then you know who the really good friends are. "
    # true class 2
    opt.document = "premenstrual syndrome (pms) is a group of symptoms related to the menstrual cycle. pms is linked to changes in the endocrine system, which produces hormones that control the menstrual cycle. medical experts don't fully understand the chain of events that causes premenstrual symptoms to be severe in some women and not in others. the one direct cause that is known to affect some women is genetic: many women with pms have a close family member with a history of pms. "
    
    # pass relevant label_map
    predict(opt,yahoo_label_map)
    print()


