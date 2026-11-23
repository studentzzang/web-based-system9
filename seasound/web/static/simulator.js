(() => {
  'use strict';

  const root = document.getElementById('sim-root');
  if (!root) return;
  const $ = id => document.getElementById(id);
  const ui = {
    date: $('sim-date'), coordinates: $('sim-coordinates'), radius: $('sim-radius'),
    radiusValue: $('sim-radius-value'), window: $('sim-window'), count: $('sim-count'),
    level: $('sim-level'), listen: $('sim-listen'), audioStatus: $('sim-audio-status'),
    play: $('sim-play'), time: $('sim-time'), clock: $('sim-clock'), status: $('sim-status'),
    graph: $('sim-graph'), graphRange: $('sim-graph-range'), vessels: $('sim-vessels'),
    sampleDataset: $('sim-sample-dataset'),
    vesselSummary: $('sim-vessel-summary'),
  };
  const colors = {
    Cargo: '#2b8b75', Tanker: '#bc7650', Passenger: '#4d85b4',
    'Tug/Work': '#8773af', Fishing: '#b39a40',
    'Small craft': '#728e9b', Other: '#879198',
  };
  let metadata, map, receiverMarker, radiusCircle, vesselLayer;
  let receiver, currentScene, sceneRequest, audioRequest, sceneEpoch = 0, audioEpoch = 0;
  let debounceTimer, playbackTimer, audioContext, currentSource, currentGain;
  let listening = false;

  function status(element, message, error = false) {
    element.textContent = message;
    element.dataset.error = error ? 'true' : 'false';
  }

  async function getJson(url, signal) {
    const response = await fetch(url, { signal });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error || `요청 실패 (${response.status})`);
    return body;
  }

  function selectedTime() {
    return new Date(Date.parse(metadata.start_utc) + Number(ui.time.value) * 60000)
      .toISOString().replace('.000Z', 'Z');
  }

  function updateClock() {
    if (metadata) ui.clock.textContent = `${selectedTime().slice(11, 16)} UTC`;
  }

  function queryUrl(base, includeWindow = false) {
    const url = new URL(base, window.location.href);
    url.searchParams.set('at', selectedTime());
    url.searchParams.set('lat', receiver.lat.toFixed(6));
    url.searchParams.set('lon', receiver.lng.toFixed(6));
    url.searchParams.set('radius_km', ui.radius.value);
    if (includeWindow) url.searchParams.set('window_minutes', ui.window.value);
    return url;
  }

  function setReceiver(point) {
    receiver = L.latLng(point.lat, point.lng);
    receiverMarker.setLatLng(receiver);
    radiusCircle.setLatLng(receiver);
    ui.coordinates.textContent = `${receiver.lat.toFixed(5)}° N · ${Math.abs(receiver.lng).toFixed(5)}° ${receiver.lng < 0 ? 'W' : 'E'}`;
    scheduleScene();
  }

  function renderMap(scene) {
    vesselLayer.clearLayers();
    for (const vessel of scene.positions) {
      const color = vessel.inside_radius ? (colors[vessel.unified_label] || colors.Other) : '#a3b2b5';
      const point = L.circleMarker([vessel.lat, vessel.lon], {
        radius: vessel.inside_radius ? 6 : 3,
        color: vessel.inside_radius ? '#fff' : '#a3b2b5', weight: 1,
        fillColor: color, fillOpacity: vessel.inside_radius ? 0.95 : 0.45,
      });
      const popup = document.createElement('div');
      popup.textContent = `${vessel.vessel_name} · ${vessel.unified_label}${vessel.inside_radius ? ' · 청음 반경 내' : ''}`;
      point.bindPopup(popup).addTo(vesselLayer);
    }
  }

  function renderVessels(scene) {
    ui.vessels.replaceChildren();
    ui.vesselSummary.textContent = scene.vessel_count
      ? `수신 ${scene.vessel_count}척 · 반경 내 선박 모두 합성`
      : '수신 0척';
    if (!scene.vessels.length) {
      const row = document.createElement('tr');
      const cell = document.createElement('td');
      cell.colSpan = 6;
      cell.className = 'sim-empty';
      cell.textContent = '이 시각과 반경에는 최근 AIS 위치가 있는 선박이 없습니다.';
      row.appendChild(cell);
      ui.vessels.appendChild(row);
      return;
    }
    for (const vessel of scene.vessels.slice(0, 20)) {
      const row = document.createElement('tr');
      const values = [vessel.vessel_name, vessel.unified_label,
        `${(vessel.distance_m / 1000).toFixed(2)} km`,
        vessel.sog_kn === null ? '자료 없음' : `${vessel.sog_kn.toFixed(1)} kn`,
        `${vessel.tl_db.toFixed(1)} dB`, `${vessel.rl_db.toFixed(1)} dB`];
      values.forEach((value, index) => {
        const cell = document.createElement('td');
        if (index === 1) {
          const dot = document.createElement('i');
          dot.className = 'sim-vessel-dot';
          dot.style.background = colors[vessel.unified_label] || colors.Other;
          cell.appendChild(dot);
        }
        cell.appendChild(document.createTextNode(value));
        row.appendChild(cell);
      });
      ui.vessels.appendChild(row);
    }
  }

  function renderGraph(scene) {
    const canvas = ui.graph;
    const ctx = canvas.getContext('2d');
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#f8fbfa';
    ctx.fillRect(0, 0, width, height);
    const series = scene.series;
    if (!series.length) return;
    const valid = series.map(point => point.total_rl_db).filter(value => value !== null);
    const left = 58, right = width - 20, top = 18, bottom = height - 36;
    const min = valid.length ? Math.floor((Math.min(...valid) - 5) / 5) * 5 : 0;
    const max = valid.length ? Math.ceil((Math.max(...valid) + 5) / 5) * 5 : 10;
    const start = Date.parse(series[0].at_utc), end = Date.parse(series.at(-1).at_utc);
    const x = at => left + (Date.parse(at) - start) / Math.max(1, end - start) * (right - left);
    const y = level => bottom - (level - min) / Math.max(1, max - min) * (bottom - top);
    ctx.font = '12px Segoe UI, sans-serif';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 3; i++) {
      const level = min + (max - min) * i / 3;
      const yy = y(level);
      ctx.strokeStyle = '#e1eae8';
      ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(right, yy); ctx.stroke();
      ctx.fillStyle = '#698087';
      ctx.fillText(`${level.toFixed(0)} dB`, left - 9, yy + 4);
    }
    ctx.strokeStyle = '#348b7d';
    ctx.lineWidth = 3;
    ctx.beginPath();
    let started = false;
    for (const point of series) {
      if (point.total_rl_db === null) { started = false; continue; }
      const xx = x(point.at_utc), yy = y(point.total_rl_db);
      if (started) ctx.lineTo(xx, yy); else ctx.moveTo(xx, yy);
      started = true;
    }
    ctx.stroke();
    const cursorX = Math.max(left, Math.min(right, x(scene.at_utc)));
    ctx.strokeStyle = '#bb7553'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(cursorX, top); ctx.lineTo(cursorX, bottom); ctx.stroke();
    ctx.fillStyle = '#6a7f86';
    ctx.textAlign = 'left'; ctx.fillText(`${series[0].at_utc.slice(11, 16)} UTC`, left, height - 11);
    ctx.textAlign = 'right'; ctx.fillText(`${series.at(-1).at_utc.slice(11, 16)} UTC`, right, height - 11);
    if (!valid.length) {
      ctx.textAlign = 'center'; ctx.fillText('선택 구간에 수신 선박이 없습니다.', width / 2, height / 2);
    }
    ui.graphRange.textContent = `${series[0].at_utc.slice(11, 16)}–${series.at(-1).at_utc.slice(11, 16)} UTC`;
    canvas.setAttribute('aria-label', `${ui.graphRange.textContent} 시간별 수신 레벨. 현재 ${scene.total_rl_db === null ? '수신 선박 없음' : `${scene.total_rl_db.toFixed(1)} dB`}`);
  }

  function renderScene(scene) {
    currentScene = scene;
    ui.count.textContent = `${scene.vessel_count}척`;
    ui.level.textContent = scene.total_rl_db === null ? '수신 없음' : `${scene.total_rl_db.toFixed(1)} dB`;
    status(ui.status, `${scene.at_utc.slice(11, 16)} UTC · 최근 ${metadata.position_max_age_minutes}분 내 AIS 위치 기준 · ${scene.frequency_hz.toLocaleString('ko-KR')} Hz 계산`);
    renderMap(scene);
    renderVessels(scene);
    renderGraph(scene);
  }

  async function loadScene() {
    if (!metadata) return;
    if (sceneRequest) sceneRequest.abort();
    const request = new AbortController();
    sceneRequest = request;
    const epoch = ++sceneEpoch;
    status(ui.status, '선박 위치와 수신 레벨을 계산하는 중입니다…');
    try {
      const scene = await getJson(queryUrl(root.dataset.sceneUrl, true), request.signal);
      if (epoch !== sceneEpoch) return;
      renderScene(scene);
      if (listening) refreshAudio();
    } catch (error) {
      if (error.name !== 'AbortError' && epoch === sceneEpoch) status(ui.status, error.message, true);
    } finally {
      if (sceneRequest === request) sceneRequest = null;
    }
  }

  function scheduleScene(immediate = false) {
    clearTimeout(debounceTimer);
    updateClock();
    debounceTimer = setTimeout(loadScene, immediate ? 0 : 150);
  }

  function stopAudio() {
    listening = false;
    audioEpoch++;
    if (audioRequest) audioRequest.abort();
    if (currentSource) { try { currentSource.stop(); } catch (_error) { /* already stopped */ } }
    currentSource = null;
    currentGain = null;
    ui.listen.textContent = '▶ 가상 청음 시작';
    status(ui.audioStatus, '청음이 중지되었습니다.');
  }

  async function refreshAudio() {
    if (!listening || !currentScene) return;
    if (audioRequest) audioRequest.abort();
    const request = new AbortController();
    audioRequest = request;
    const epoch = ++audioEpoch;
    status(ui.audioStatus, '현재 위치의 음향을 합성하는 중입니다…');
    try {
      const response = await fetch(queryUrl(root.dataset.audioUrl, true), { signal: request.signal });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.error || `음향 생성 실패 (${response.status})`);
      }
      const buffer = await audioContext.decodeAudioData(await response.arrayBuffer());
      if (!listening || epoch !== audioEpoch) return;
      const source = audioContext.createBufferSource();
      const gain = audioContext.createGain();
      source.buffer = buffer;
      source.loop = true;
      source.connect(gain).connect(audioContext.destination);
      const now = audioContext.currentTime;
      gain.gain.setValueAtTime(0, now);
      gain.gain.linearRampToValueAtTime(1, now + 0.25);
      source.start();
      if (currentSource && currentGain) {
        currentGain.gain.cancelScheduledValues(now);
        currentGain.gain.setValueAtTime(currentGain.gain.value, now);
        currentGain.gain.linearRampToValueAtTime(0, now + 0.25);
        currentSource.stop(now + 0.3);
      }
      currentSource = source;
      currentGain = gain;
      status(ui.audioStatus, currentScene.vessel_count
        ? `${buffer.duration.toFixed(0)}초 음향 반복 재생 · 반경 내 ${currentScene.vessel_count}척 반영`
        : '반경 안에 선박이 없어 무음이 재생됩니다.');
    } catch (error) {
      if (error.name !== 'AbortError' && epoch === audioEpoch) status(ui.audioStatus, error.message, true);
    } finally {
      if (audioRequest === request) audioRequest = null;
    }
  }

  async function toggleAudio() {
    if (listening) { stopAudio(); return; }
    const Context = window.AudioContext || window.webkitAudioContext;
    if (!Context) { status(ui.audioStatus, '이 브라우저는 Web Audio를 지원하지 않습니다.', true); return; }
    try {
      audioContext ||= new Context();
      await audioContext.resume();
      listening = true;
      ui.listen.textContent = '■ 가상 청음 중지';
      refreshAudio();
    } catch (error) {
      status(ui.audioStatus, `오디오를 시작하지 못했습니다: ${error.message}`, true);
    }
  }

  function stopPlayback() {
    if (playbackTimer) clearInterval(playbackTimer);
    playbackTimer = null;
    ui.play.textContent = '▶ 시간 재생';
    ui.play.setAttribute('aria-label', 'AIS 시간 재생');
  }

  function togglePlayback() {
    if (playbackTimer) { stopPlayback(); return; }
    if (Number(ui.time.value) >= Number(ui.time.max)) ui.time.value = 0;
    ui.play.textContent = '❚❚ 일시정지';
    ui.play.setAttribute('aria-label', 'AIS 시간 재생 일시정지');
    scheduleScene(true);
    playbackTimer = setInterval(() => {
      const next = Math.min(Number(ui.time.max), Number(ui.time.value) + 5);
      ui.time.value = String(next);
      scheduleScene(true);
      if (next >= Number(ui.time.max)) stopPlayback();
    }, 2500);
  }

  async function initialize() {
    if (!window.L) { status(ui.status, '지도 라이브러리를 불러오지 못했습니다. 인터넷 연결을 확인하세요.', true); return; }
    try {
      metadata = await getJson(root.dataset.metaUrl);
      ui.date.textContent = `${metadata.date_utc} · UTC · AIS ${metadata.vessel_count}척`;
      ui.sampleDataset.textContent = metadata.sample_dataset;
      ui.time.max = String(Math.floor((Date.parse(metadata.end_utc) - Date.parse(metadata.start_utc)) / 60000));
      ui.time.value = String(Math.min(720, Number(ui.time.max)));
      ui.radius.value = String(metadata.default_radius_km);
      ui.radiusValue.textContent = `${Number(ui.radius.value).toFixed(1)} km`;
      map = L.map('sim-map', { preferCanvas: true }).setView(
        [metadata.default_receiver.lat, metadata.default_receiver.lon], 10);
      L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 18,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      }).addTo(map);
      L.control.scale({ imperial: false }).addTo(map);
      vesselLayer = L.layerGroup().addTo(map);
      receiver = L.latLng(metadata.default_receiver.lat, metadata.default_receiver.lon);
      radiusCircle = L.circle(receiver, { radius: Number(ui.radius.value) * 1000,
        color: '#2a887d', weight: 2, fillColor: '#5eb3a1', fillOpacity: 0.11 }).addTo(map);
      receiverMarker = L.marker(receiver, { draggable: true, icon: L.divIcon({
        className: '', html: '<span class="sim-receiver-icon" aria-hidden="true"></span>',
        iconSize: [22, 22], iconAnchor: [11, 11],
      }) }).addTo(map).bindTooltip('가상 수중청음기 · 드래그하여 이동');
      map.on('click', event => setReceiver(event.latlng));
      receiverMarker.on('drag', event => radiusCircle.setLatLng(event.target.getLatLng()));
      receiverMarker.on('dragend', event => setReceiver(event.target.getLatLng()));
      ui.coordinates.textContent = `${receiver.lat.toFixed(5)}° N · ${Math.abs(receiver.lng).toFixed(5)}° W`;
      for (const control of [ui.radius, ui.window, ui.listen, ui.play, ui.time]) control.disabled = false;
      updateClock();
      await loadScene();
    } catch (error) {
      status(ui.status, error.message, true);
    }
  }

  ui.radius.addEventListener('input', () => {
    ui.radiusValue.textContent = `${Number(ui.radius.value).toFixed(1)} km`;
    if (radiusCircle) radiusCircle.setRadius(Number(ui.radius.value) * 1000);
    scheduleScene();
  });
  ui.window.addEventListener('change', () => scheduleScene(true));
  ui.time.addEventListener('input', () => { stopPlayback(); scheduleScene(); });
  ui.time.addEventListener('change', () => scheduleScene(true));
  ui.play.addEventListener('click', togglePlayback);
  ui.listen.addEventListener('click', toggleAudio);
  window.addEventListener('pagehide', () => { stopPlayback(); stopAudio(); });
  initialize();
})();
