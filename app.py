from flask import Flask, render_template, Response, request
import cv2
from load_data import *
from predict import *
import webbrowser
import subprocess
from googleapiclient.discovery import build
import sqlite3
suggestions_list=[]
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = Flask(__name__)
video = cv2.VideoCapture(0)
root=[]

from bs4 import BeautifulSoup
import requests
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

def weather(city):
    city = city.replace(" ", "+")
    res = requests.get(
        f'https://www.google.com/search?q={city}&oq={city}&aqs=chrome.0.35i39l2j0l4j46j69i60.6128j1j7&sourceid=chrome&ie=UTF-8', headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')
    location = soup.select('#wob_loc')[0].getText().strip()
    time = soup.select('#wob_dts')[0].getText().strip()
    info = soup.select('#wob_dc')[0].getText().strip()
    weather = soup.select('#wob_tm')[0].getText().strip()
    return info



@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


@app.route('/suggestions')
def suggestions():
    text = request.args.get('jsdata')
    text=text.split('|')
    youtube = build('youtube','v3', developerKey="AIzaSyBTZN1W54ksvEl37jHwW4kdJ75vBh89vnA")
# retrieve youtube video results
    str=text[0].split('=')
    video_request=youtube.videos().list(
    part='snippet,statistics',
    id=str[1]
    )
    video_response = video_request.execute()
    title = video_response['items'][0]['snippet']['title']
    text.append(title)
    suggestions_list.append(text)
    print(suggestions_list)
    return render_template('suggestions.html', suggestions=suggestions_list)

@app.route('/takeimage', methods = ['POST'])
def takeimage():
    name = request.form['name']
    #print(name)
    _, frame = video.read()
    cv2.imwrite('user.jpg', frame)
    user_mood=main_data_of_face('user.jpg')
    print(user_mood)
    if(user_mood=="neutral"):
        city = "Dindugal weather"
        if("sunny" in weather(city).lower()):
            user_link=link_give(weather(city)+' songs')
            print(weather(city))
        else:
            user_link=link_give(str(user_mood)+' songs')
    else:
        user_link=link_give(str(user_mood)+' songs')
    print(user_link)
    root.append(user_mood)
    url = user_link
    chrome_path = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
    first_url = user_link
    command = f"cmd /C \"{chrome_path}\" {first_url} --new-window"
# Alternative way that achieves the same result
# command = f"cmd /C start chrome {first_url} --new-window"

    subprocess.Popen(command)
    """
    webbrowser.register('chrome',
	None,
	webbrowser.BackgroundBrowser("chrome.exe"))
    webbrowser.get('chrome').open(url)
    """
    x=user_link+'|'+user_mood

    return x

def gen():
    """Video streaming generator function."""
    while True:
        rval, frame = video.read()
        cv2.imwrite('t.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()
    app.debug = True
