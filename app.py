from flask import Flask, render_template, request, url_for
import os

app = Flask(
    __name__,
    static_url_path=''
)

@app.route('/')
def index():
    return render_template('index.html')

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
    
# @app.route('/training')
# def training():
#     relationExtraction.training()
#     return 'success'
# @app.route('/process', methods=['GET'])
# def process():
#     return relationExtraction.processInput(request.args.get('text'))
