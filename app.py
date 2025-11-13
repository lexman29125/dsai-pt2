from flask import Flask, render_template, request
import joblib
import os
from groq import Groq

from dotenv import load_dotenv
if os.path.exists('.env'):
    load_dotenv()

#os.environ["GROQ_API_KEY"] = os.environ.get('GROQ_API_KEY')

client = Groq()

application = Flask(__name__)

@application.route("/",methods=["GET","POST"])
def index():
    return(render_template("index.html"))

@application.route("/main",methods=["GET","POST"])
def main():
    return(render_template("main.html"))

@application.route("/dbs",methods=["GET","POST"])
def dbs():
    return(render_template("dbs.html"))


@application.route("/dbs_prediction",methods=["GET","POST"])
def dbs_prediction():
    q = float(request.form.get("q"))
    model = joblib.load("DBS_SGD_model.pkl")
    r = model.predict([[q]])
    return(render_template("dbs_prediction.html",r=r))

@application.route("/chatbot",methods=["GET","POST"])
def chatbot():
    return(render_template("chatbot.html"))

@application.route("/llama",methods=["GET","POST"])
def llama():
    return(render_template("llama.html"))

@application.route("/llama_result",methods=["GET","POST"])
def llama_result():
    q = request.form.get("q")
    r = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": q}])
    r = r.choices[0].message.content
    return(render_template("llama_result.html",r=r))

if __name__ == "__main__":
    application.run()
