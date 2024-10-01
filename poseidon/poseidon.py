# zeus.py

import random
import numpy
import tflearn
import tensorflow
import json
import pickle
from flask import Flask, request, jsonify
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

