from flask import Flask, jsonify ,request

app = Flask(__name__)


@app.route('/get')
def ans():
    data =request.get_json()
    print(data)
if __name__ == '__main__':
    app.run(host='192.168.80.87', port=5000)
