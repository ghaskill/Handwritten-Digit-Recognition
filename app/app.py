from flask import Flask, request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__, template_folder='templates')

model = pickle.load(open('knn_model_pickle.sav', 'rb'))

@app.route('/')
def main():
  return render_template("main.html")

@app.route('/hook', methods=['POST'])
def get_image():
  img = request.values['img']
  print ('Image Received')
  
if __name__ == '__main__':
  app.run()
