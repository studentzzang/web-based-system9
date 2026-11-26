'use strict';
const $=id=>document.getElementById(id), C=SeaSound;
// Flask renders the registered paths on the form. Older Explorer pages can
// derive the API prefix from the mounted /explorer page.
const routeFields = $('search-form').dataset;
const explorerPage = new URL(location.href);
explorerPage.pathname = explorerPage.pathname.replace(/\/+$/, '');
const apiBase = new URL('api/', explorerPage);
const routes = {
  recordings: routeFields.listEndpoint,
  'recordings/audio': routeFields.audioEndpoint,
  'recordings/waveform': routeFields.waveformEndpoint,
  'recordings/analysis': routeFields.analysisEndpoint,
  'recordings/spectrogram': routeFields.spectrogramEndpoint,
  stft: routeFields.stftEndpoint,
};
function api(name, params = {}) {
  const url = routes[name] ? new URL(routes[name], location.origin) : new URL(name, apiBase);
  for (const [key, value] of Object.entries(params)) {
    if (value !== '' && value != null) url.searchParams.set(key, value);
  }
  return url;
}
const message=(target,text,error=false)=>{target.textContent=text;target.dataset.error=String(error);};
async function json(url,signal){const response=await fetch(url,{signal,headers:{Accept:'application/json'}});let raw;try{raw=await response.json();}catch{throw new Error(`JSON 응답을 받지 못했습니다 (HTTP ${response.status}).`);}if(!response.ok)throw new Error(raw.error || `HTTP ${response.status}`);return raw;}
const RANGE_IDS = ['fmin', 'fmax', 'dbmin', 'dbmax'];
const MODES = {
  stft: {label: 'STFT', axis: '주파수 (Hz)', ranges: [0, 8000, -80, 0]},
  logmel: {label: 'Log-Mel', axis: '멜 대역 중심 (Hz)', ranges: [10, 8000, -80, 0]},
  lofar: {label: 'LOFAR', axis: '주파수 (Hz)', ranges: [0, 200, -80, 0]},
  demon: {label: 'DEMON', axis: '변조 주파수 (Hz)', ranges: [0, 50, -80, 0]},
};
const previousRanges = new Map();
let activeMode = 'stft';
const currentRanges = () => C.ranges(RANGE_IDS.map(id => $(id).value));
function clear(canvas,label){canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);canvas.setAttribute('aria-label',label);}
const palette = Array.from({length: 256}, (_, i) => [...C.palette(i / 255), 255]);

function drawSpectrum(canvas, data, range, mode, duration) {
  const {lo, hi, dlo, dhi} = range;
  if (!data.f.some(f => f >= lo && f <= hi)) {
    throw new Error('선택 범위에 주파수 데이터가 없습니다.');
  }
  const ctx = canvas.getContext('2d');
  const left = 80, top = 20, width = 890, height = 310;
  const image = ctx.createImageData(width, height);
  const frequencyAt = fraction => mode === 'logmel'
    ? C.fromMel(C.toMel(hi) - fraction * (C.toMel(hi) - C.toMel(lo)))
    : hi - fraction * (hi - lo);
  const axis = MODES[mode].axis;
  const cols = Array.from({length: width}, (_, x) => C.nearest(data.t, (x + .5) / width * duration));
  for (let y = 0; y < height; y++) {
    const frequency = frequencyAt((y + .5) / height);
    const dataRow = data.z[C.nearest(data.f, frequency)];
    const outside = frequency < data.f[0] || frequency > data.f.at(-1);
    for (let x = 0; x < width; x++) {
      const index = Math.round(255 * C.clamp((dataRow[cols[x]] - dlo) / (dhi - dlo)));
      image.data.set(outside ? [15, 25, 40, 255] : palette[index], (y * width + x) * 4);
    }
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.putImageData(image, left, top);
  ctx.font = '16px sans-serif';
  ctx.fillStyle = '#e4eeee';
  for (let i = 0; i <= 4; i++) {
    ctx.textAlign = 'right';
    ctx.fillText(Number(frequencyAt(i / 4).toFixed(2)), left - 8, top + i / 4 * height + 5);
    ctx.textAlign = 'center';
    ctx.fillText((i / 4 * duration).toFixed(2), left + i / 4 * width, top + height + 25);
  }
  ctx.fillText('시간 (s)', left + width / 2, 390);
  ctx.save();
  ctx.translate(20, top + height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(axis, 0, 0);
  ctx.restore();
  canvas.setAttribute('aria-label', MODES[mode].label + ' · ' + axis +
    ' ' + lo + '–' + hi + ' · ' + dlo + '–' + dhi + ' 상대 dB');
}

function drawWave(canvas,wave){
  const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.strokeStyle='#3c8b78';ctx.lineWidth=1;ctx.beginPath();
  const scale=Math.max(1e-9,...wave.min.map(Math.abs),...wave.max.map(Math.abs));
  wave.times.forEach((t,i)=>{const x=t/wave.duration*canvas.width;ctx.moveTo(x,80-wave.max[i]/scale*70);ctx.lineTo(x,80-wave.min[i]/scale*70);});
  ctx.stroke();ctx.strokeStyle='#b6c9c4';ctx.beginPath();ctx.moveTo(0,80);ctx.lineTo(1000,80);ctx.stroke();
  canvas.setAttribute('aria-label',`전체 ${wave.duration.toFixed(2)}초 파형, 채널 평균. 클릭 또는 좌우 방향키로 탐색`);
}
const panels=[];
class RecordingPanel {
  constructor(slot){
    this.slot=slot;this.path='';this.spectrum=null;this.plotVisible=false;this.wave=null;this.epoch=0;this.analysisEpoch=0;this.controller=null;this.analysisController=null;this.raf=0;
    this.spectra = new Map();
    this.spectrumController = null;
    this.audio=this.get('audio');
    this.get('load').addEventListener('submit',e=>{e.preventDefault();this.load(this.get('path').value.trim());});
    this.audio.addEventListener('play', () => this.animate());
    for(const name of ['timeupdate','seeking','seeked','loadedmetadata','pause','ended'])this.audio.addEventListener(name,()=>this.updateCursor());
    this.audio.addEventListener('loadedmetadata',()=>message(this.get('audio-status'),'재생 준비 완료 · 원본 길이 유지, PCM16 WAV 제공'));
    this.audio.addEventListener('error',()=>{if(this.audio.getAttribute('src'))message(this.get('audio-status'),'오디오를 재생하지 못했습니다. 녹음 경로와 서버 응답을 확인하세요.',true);});
    for(const type of ['wave','spectrogram']){
      const canvas=this.get(type);
      canvas.addEventListener('click',event=>{
        const rect=canvas.getBoundingClientRect(), fraction=(event.clientX-rect.left)/rect.width;
        if(type==='wave')this.seek(C.seek(fraction,this.wave?.duration || this.audio.duration));
        else if(this.spectrum && fraction>=.08 && fraction<=.97)this.seek(C.seek((fraction-.08)/.89, this.plotDuration()));
      });
      canvas.addEventListener('keydown',event=>{if(['ArrowLeft','ArrowRight'].includes(event.key)){event.preventDefault();this.seek(this.audio.currentTime+(event.key==='ArrowRight'?1:-1));}});
    }
  }
  get(name){return $(`${name}-${this.slot}`);}
  resetMetrics() {
    this.get('tonal').textContent = '—';
    this.get('shaft').textContent = '—';
    this.get('shaft-evidence').replaceChildren();
    this.get('shaft-evidence').hidden = true;
  }
  plotDuration() {
    return this.spectrum?.duration ?? this.wave?.duration ??
      (Number.isFinite(this.audio.duration) ? this.audio.duration : this.spectrum?.t.at(-1)) ?? 0;
  }
  updateCursor(){
    const time=this.audio.currentTime || 0;this.get('time').textContent=`${time.toFixed(2)} s`;
    const wc=this.get('wave-cursor'),sc=this.get('spec-cursor');
    wc.hidden=!this.wave;if(this.wave)wc.style.left=`${C.clamp(time/this.wave.duration)*100}%`;
    sc.hidden=!this.spectrum || !this.plotVisible;
    if (this.spectrum) sc.style.left = (8 + 89 * C.clamp(time / (this.plotDuration() || 1))) + '%';
  }
  animate(){cancelAnimationFrame(this.raf);const tick=()=>{this.updateCursor();if(!this.audio.paused && !this.audio.ended)this.raf=requestAnimationFrame(tick);};tick();}
  seek(time){if(Number.isFinite(time) && Number.isFinite(this.audio.duration))this.audio.currentTime=Math.max(0,Math.min(this.audio.duration,time));this.updateCursor();}
  async load(path,record=null){
    if(!path)return;
    const epoch=++this.epoch;this.controller?.abort();this.analysisController?.abort();this.analysisEpoch++;this.controller=new AbortController();
    const signal=this.controller.signal;this.path=path;this.spectrum=null;this.plotVisible=false;this.wave=null;this.resetMetrics();
    this.spectra.clear();
    this.audio.pause();cancelAnimationFrame(this.raf);this.audio.removeAttribute('src');this.audio.load();
    this.get('path').value=path;
    this.get('meta').textContent=record?`${record.dataset} · ${record.class} · ${record.duration.toFixed(2)}초 · ${record.sample_rate} Hz · ${record.channels}채널`:`경로: ${path}`;
    clear(this.get('wave'),'파형을 불러오는 중');this.updateCursor();
    message(this.get('audio-status'),'오디오를 준비하는 중입니다…');this.audio.src=api('recordings/audio',{path}).href;
    const timeout=setTimeout(()=>this.controller?.signal===signal && this.controller.abort(),60000);
    const task=async(name,url,success)=>{
      message(this.get(`${name}-status`),'불러오는 중입니다…');
      try{const raw=await json(url,signal);if(epoch!==this.epoch)return;success(raw);}
      catch(error){if(epoch!==this.epoch)return;message(this.get(`${name}-status`),error.name==='AbortError'?'요청 시간이 초과되었거나 취소되었습니다. 다시 불러오세요.':error.message,true);}
    };
    const jobs=[
      this.loadSpectrum(),
      task('wave',api('recordings/waveform',{path}),raw=>{this.wave=raw;drawWave(this.get('wave'),raw);message(this.get('wave-status'),`전체 ${raw.duration.toFixed(2)}초 · 채널 평균 파형 · 진폭은 이 파형 내 최댓값 기준`);this.redraw();this.updateCursor();}),
      this.analyze()
    ];
    await Promise.allSettled(jobs);clearTimeout(timeout);
  }
  async loadSpectrum() {
    this.spectrumController?.abort();
    this.spectrum = null;
    this.plotVisible = false;
    const mode = activeMode;
    this.get('spectrum-label').textContent = MODES[mode].label + ' · 전체 녹음';
    const target = this.get('stft-status');
    if (!this.path) {
      clear(this.get('spectrogram'), MODES[mode].label + ' 데이터 없음');
      this.updateCursor();
      return;
    }
    if (this.spectra.has(mode)) {
      this.spectrum = this.spectra.get(mode);
      this.redraw();
      return;
    }
    const controller = new AbortController();
    this.spectrumController = controller;
    clear(this.get('spectrogram'), MODES[mode].label + ' 불러오는 중');
    this.updateCursor();
    message(target, '불러오는 중입니다…');
    const timeout = setTimeout(() => controller.abort(), 120000);
    try {
      const params = {path: this.path, mode, detail: mode === 'stft' ? 'frequency' : ''};
      const raw = await json(api(mode === 'stft' ? 'stft' : 'recordings/spectrogram', params), controller.signal);
      if (controller !== this.spectrumController || controller.signal.aborted) return;
      this.spectrum = C.normalize(raw);
      this.spectra.set(mode, this.spectrum);
      this.redraw();
    } catch (error) {
      if (controller === this.spectrumController) {
        message(target, error.name === 'AbortError' ? '그래프 요청이 취소됐거나 시간이 초과됐습니다.' : error.message, true);
      }
    } finally {
      clearTimeout(timeout);
    }
  }
  redraw(){
    if(!this.spectrum)return;
    try{
      const range=currentRanges();drawSpectrum(this.get('spectrogram'), this.spectrum, range, activeMode, this.plotDuration());this.plotVisible=true;
      const panel=this.get('panel');panel.querySelector('.color-low').textContent=`${range.dlo} dB`;panel.querySelector('.color-high').textContent=`${range.dhi} dB`;
      message(this.get('stft-status'),`${this.spectrum.f.length}주파수 × ${this.spectrum.t.length}프레임 · 각 녹음 최대 파워 기준`);
    }catch(error){this.plotVisible=false;clear(this.get('spectrogram'),'표시 범위 오류');message(this.get('stft-status'),error.message,true);}this.updateCursor();
  }
  showShaftResult(result) {
    const shaft = this.get('shaft');
    if (result.background) {
      shaft.textContent = '해당 없음';
      return;
    }

    const estimate = result.shafts[0];
    const reference = result.rpm_reference;
    const evidence = this.get('shaft-evidence');
    const entries = [];
    if (estimate) {
      const hasBlades = result.shaft_method === 'explicit_blade_pass';
      shaft.textContent = '약 ' + estimate.rpm.toFixed(2) + ' rpm 추정';
      entries.push(hasBlades
        ? '날개 통과 주파수 가정 · ' + estimate.blade_pass_hz.toFixed(3) + ' Hz ÷ ' + estimate.blade_count + ' × 60'
        : '기본 반복을 축 1회전으로 가정 · ' + estimate.hz.toFixed(3) + ' Hz × 60');
    } else if (reference) {
      shaft.textContent = '회전수 추정 보류';
      entries.push('축 1회전으로 가정한 환산 참고값 · 약 ' + reference.rpm.toFixed(2) +
        ' rpm (' + reference.hz.toFixed(3) + ' Hz × 60)');
      if (reference.kind === 'dominant') {
        entries.push('우세 변조 ' + reference.hz.toFixed(3) + ' Hz · ' +
          reference.segments + '/' + result.segments + '구간 · 배음 조합 근거 부족');
      }
    } else {
      shaft.textContent = '회전수 추정 보류';
      entries.push('환산할 반복 주파수 검출 없음');
    }

    for (const candidate of result.shaft_candidates ?? []) {
      const run = candidate.continuity;
      let text = candidate.hz.toFixed(3) + ' Hz 반복 후보 · 전체 ' +
        candidate.segments + '/' + result.segments + '구간 · 배음 차수 ' +
        candidate.orders.join('·');
      if (run?.windows) {
        text += ' · ' + run.start.toFixed(1) + '–' + run.end.toFixed(1) +
          '초, 배음 근거 ' + run.windows + '구간 연속';
      }
      entries.push(text);
    }
    evidence.hidden = entries.length === 0;
    for (const entry of entries) {
      const item = document.createElement('li');
      item.textContent = entry;
      evidence.append(item);
    }
  }

  async analyze(){
    if(!this.path){message(this.get('analysis-status'),'먼저 녹음을 선택하세요.',true);return;}
    this.analysisController?.abort();const controller=new AbortController();this.analysisController=controller;
    const token=++this.analysisEpoch,epoch=this.epoch;this.resetMetrics();
    message(this.get('analysis-status'),'분석 중입니다…');
    const params = {path: this.path, scope: 'clip'};
    const timeout=setTimeout(()=>controller.abort(),120000);
    try{
      const r=await json(api('recordings/analysis',params),controller.signal);
      if(token!==this.analysisEpoch || epoch!==this.epoch)return;
      const tonal = r.tonals[0];
      this.get('tonal').textContent = tonal
        ? tonal.hz.toFixed(2) + ' Hz · ' + tonal.segments + '/' + r.segments + '구간에서 검출'
        : '검출 없음';
      this.showShaftResult(r);
      message(this.get('analysis-status'), '전체 ' + r.end.toFixed(1) +
        '초 · ' + r.segment_seconds + '초 창 ' + r.segments + '개 구간 분석' +
        ' · 토널 분석 ' + r.tonal_band_hz.join('–') + ' Hz');
    }catch(error){if(token===this.analysisEpoch && epoch===this.epoch)message(this.get('analysis-status'),error.name==='AbortError'?'분석 요청이 취소되었거나 시간이 초과되었습니다.':error.message,true);}
    finally{clearTimeout(timeout);}
  }
}
for(const slot of ['a','b'])panels.push(new RecordingPanel(slot));
const modeButtons = document.querySelectorAll('[data-spectrum-mode]');
for (const button of modeButtons) {
  button.addEventListener('click', () => {
    const next = button.dataset.spectrumMode;
    if (next === activeMode) return;
    previousRanges.set(activeMode, RANGE_IDS.map(id => $(id).value));
    activeMode = next;
    const range = previousRanges.get(next) ?? MODES[next].ranges;
    RANGE_IDS.forEach((id, index) => { $(id).value = range[index]; });
    for (const option of modeButtons) option.setAttribute('aria-pressed', String(option === button));
    $('range-axis').textContent = MODES[next].axis + ' 표시 범위';
    message($('display-status'), MODES[next].label + ' · 각 녹음과 모드의 최대 파워 기준');
    panels.forEach(panel => panel.loadSpectrum());
  });
}
$('redraw').addEventListener('click',()=>{try{currentRanges();panels.forEach(p=>p.redraw());message($('display-status'),'표시 범위를 적용했습니다.');}catch(error){message($('display-status'),error.message,true);}});
$('full-range').addEventListener('click',()=>{const loaded=panels.map(p=>p.spectrum).filter(Boolean);if(!loaded.length){message($('display-status'),'먼저 녹음을 불러오세요.',true);return;}$('fmin').value=Math.min(...loaded.map(d=>d.f[0]));const hi=Math.max(...loaded.map(d=>d.f.at(-1)));$('fmax').value=Math.max(hi,Number($('fmin').value)+1);$('redraw').click();});
let offset=0,total=0,searchController=null,searchEpoch=0;
function options(select,values,label){const current=select.value;select.replaceChildren(new Option(label,''),...values.map(v=>new Option(v,v)));select.value=values.includes(current)?current:'';}
async function search(refresh=false){
  searchController?.abort();const controller=new AbortController();searchController=controller;const token=++searchEpoch;
  const timeout=setTimeout(()=>controller.abort(),30000);
  message($('search-status'),'녹음 목록을 검색하는 중입니다…');
  try{
    const r=await json(api('recordings',{dataset:$('dataset').value,class:$('category').value,q:$('query').value,min_duration:$('duration-min').value,max_duration:$('duration-max').value,offset,limit:20,refresh:refresh?'1':''}),controller.signal);
    if(token!==searchEpoch)return;
    total=r.total;options($('dataset'),r.datasets,'전체 데이터셋');options($('category'),r.classes,'전체 클래스');$('recordings').replaceChildren();
    for(const record of r.recordings){
      const row=document.createElement('tr');
      for(const text of [`${record.name}\n${record.path}`,`${record.dataset} / ${record.class}`,`${record.duration.toFixed(2)}초`]){const cell=document.createElement('td');cell.textContent=text;row.append(cell);}
      const actions=document.createElement('td');
      panels.forEach(panel=>{const button=document.createElement('button');button.type='button';button.textContent=`${panel.slot.toUpperCase()}에 담기`;button.addEventListener('click',()=>panel.load(record.path,record));actions.append(button);});row.append(actions);$('recordings').append(row);
    }
    message($('search-status'),`${total}개 검색됨. ${r.notices.join(' ')}`);
    $('page-info').textContent=total?`${offset+1}–${Math.min(offset+20,total)} / ${total}`:'검색 결과 없음';
    $('previous').disabled=offset===0;$('next').disabled=offset+20>=total;
  }catch(error){if(token===searchEpoch){$('recordings').replaceChildren();$('previous').disabled=true;$('next').disabled=true;$('page-info').textContent='';message($('search-status'),error.name==='AbortError'?'검색 시간이 초과되었습니다. 다시 시도하세요.':error.message,true);}}
  finally{clearTimeout(timeout);}
}
$('search-form').addEventListener('submit',event=>{event.preventDefault();offset=0;search();});
$('dataset').addEventListener('change',()=>{$('category').value='';offset=0;search();});
$('category').addEventListener('change',()=>{offset=0;search();});
$('refresh').addEventListener('click',()=>{offset=0;search(true);});
$('previous').addEventListener('click',()=>{offset=Math.max(0,offset-20);search();});
$('next').addEventListener('click',()=>{if(offset+20<total){offset+=20;search();}});
search();
