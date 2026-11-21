import os
from pathlib import Path

from flask import Flask, render_template
from dsp.stft_cache import WAV_ROOT
from web.stft_api import stft_bp
from web.explorer_api import explorer_bp
from web.traffic_api import traffic_bp
from web.simulator_api import simulator_bp

app = Flask(__name__)
app.register_blueprint(stft_bp)
app.config.update(
    AUDIO_ROOT=os.environ.get('SEASOUND_AUDIO_ROOT', str(WAV_ROOT)),
    RECORDINGS_CSV=os.environ.get('SEASOUND_RECORDINGS_CSV', str(Path(__file__).resolve().parents[1] / 'data' / 'catalog.csv')),
    AIS_CSV=os.environ.get('SEASOUND_AIS_CSV', str(Path(__file__).resolve().parents[1] / 'data' / 'ais_clean.csv')),
    SIMULATOR_CONFIG=os.environ.get('SEASOUND_SIMULATOR_CONFIG', str(Path(__file__).resolve().parents[1] / 'configs' / 'simulator.yaml')),
    SIMULATOR_RUNS_ROOT=os.environ.get('SEASOUND_SIMULATOR_RUNS_ROOT', str(Path(__file__).resolve().parents[1] / 'runs')),
    AUDIO_PLAYBACK_CACHE=str(Path(app.instance_path) / 'playback'),
)
app.register_blueprint(explorer_bp)
app.register_blueprint(traffic_bp)
app.register_blueprint(simulator_bp)


@app.get('/')
def index():
    return render_template('index.html', page='index', title='프로젝트 홈')


@app.get('/explorer')
def explorer():
    return render_template('explorer.html', page='explorer', title='음향 분석')


@app.get('/classifier')
def classifier():
    return render_template('classifier.html', page='classifier', title='AI 선종 분류')


@app.get('/traffic')
def traffic():
    return render_template('traffic.html', page='traffic', title='AIS 항적 지도')


@app.get('/simulator')
def simulator():
    return render_template('simulator.html', page='simulator', title='가상 수중청음기')


if __name__ == '__main__':
    app.run(debug=True)
