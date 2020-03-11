"""This script is used define server routing."""

import os
from flask import Flask, render_template, request, url_for
from flask_jsglue import JSGlue
from watermarking import process as p

PROCESS = p.Process()
APP = Flask(
    __name__,
    static_url_path=''
)
JSGLUE = JSGlue(APP)

@APP.route('/')
def index():
    """Return client main view."""
    return render_template('index.html')

@APP.route('/input/host', methods=['POST'])
def store_host():
    """Keep host image and return its preview."""
    return PROCESS.get_preview_host(request.data)

@APP.route('/input/wm', methods=['POST'])
def store_watermark():
    """Keep watermark image and return its preview."""
    return PROCESS.get_preview_watermark(request.data)

@APP.route('/input/wmed', methods=['POST'])
def store_watermarked():
    """Keep watermarked image and return its preview."""
    return PROCESS.get_preview_watermarked(request.data)

@APP.route('/embed', methods=['POST'])
def embed_watermark_to_host():
    """Return embedded watermark image."""
    return PROCESS.embed()

@APP.route('/extract', methods=['POST'])
def extract_watermark():
    """Return extracted watermark."""
    PROCESS.extract()
    return render_template('index.html')

@APP.context_processor
def override_url_for():
    """Override url_for function."""
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    """Override url_for function for opening static file."""
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(
                APP.root_path,
                endpoint, filename
            )
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)
    