from flask import Blueprint, Flask, render_template,request,flash,redirect,session,url_for
views = Blueprint('views',__name__)

@views.route('/', methods=['POST','GET'])
def home():
    
    return render_template("base.html")