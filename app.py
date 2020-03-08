from flask import Flask, render_template, request, url_for
from flask_jsglue import JSGlue
from watermarking import process as p
import os

process = p.Process()
app = Flask(
    __name__,
    static_url_path=''
)
jsglue = JSGlue(app)

@app.route('/')
def index():
    return render_template('index.html')

# save image to process and give back preview

@app.route('/input/host', methods=['POST'])
def storeHost():
    return process.getPreviewHost(request.data)

@app.route('/input/wm', methods=['POST'])
def storeWatermark():
    return process.getPreviewWM(request.data)

@app.route('/input/wmed', methods=['POST'])
def storeWatermarked():
    return process.getPreviewWMED(request.data)

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                 endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)
