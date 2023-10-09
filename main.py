# from asgiref import simple_server
from flask import Flask, request, render_template, Response
from flask_cors import CORS, cross_origin

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"