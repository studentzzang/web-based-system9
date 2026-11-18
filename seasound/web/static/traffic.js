(() => {
  'use strict';

  const root = document.getElementById('traffic-root');
  if (!root) return;

  const $ = id => document.getElementById(id);
  const controls = {
    filter: $('traffic-class'), fit: $('traffic-fit'), count: $('traffic-count'),
    slider: $('traffic-slider'), play: $('traffic-play'), speed: $('traffic-speed'),
    clock: $('traffic-clock'), status: $('traffic-status'), date: $('traffic-date'),
  };
  const vesselFields = {
    heading: $('traffic-vessel-heading'), empty: $('traffic-vessel-empty'),
    data: $('traffic-vessel-data'), vesselClass: $('traffic-vessel-class'),
    mmsi: $('traffic-vessel-mmsi'), sog: $('traffic-vessel-sog'),
    cog: $('traffic-vessel-cog'), lat: $('traffic-vessel-lat'),
    lon: $('traffic-vessel-lon'), time: $('traffic-vessel-time'),
    note: $('traffic-vessel-note'),
  };
  const colors = {
    Cargo: '#2b8b75', Tanker: '#bc7650', Passenger: '#4d85b4',
    'Tug/Work': '#8773af', Fishing: '#b39a40',
    'Small craft': '#728e9b', Other: '#879198',
  };

  let map;
  let markersLayer;
  let trailsLayer;
  let selectedLayer;
  let metadata;
  let currentPositions = new Map();
  let selectedMmsi = '';
  let selectedTrack = null;
  let playbackTimer = null;
  let pendingRequest = null;
  let debounceTimer = null;
  let sceneSequence = 0;

  function setStatus(message, isError = false) {
    controls.status.textContent = message;
    controls.status.dataset.error = isError ? 'true' : 'false';
  }

  async function getJson(url, signal) {
    const response = await fetch(url, { signal });
    let body;
    try {
      body = await response.json();
    } catch (_error) {
      throw new Error(`서버가 올바른 응답을 보내지 않았습니다. (${response.status})`);
    }
    if (!response.ok) throw new Error(body.error || `요청 실패 (${response.status})`);
    return body;
  }

  function selectedIsoTime() {
    return new Date(Date.parse(metadata.start_utc) + Number(controls.slider.value) * 60000)
      .toISOString().replace('.000Z', 'Z');
  }

  function updateClock() {
    if (!metadata) return;
    controls.clock.textContent = `${selectedIsoTime().slice(11, 16)} UTC`;
  }

  function formatCoordinate(value, positive, negative) {
    return `${Math.abs(value).toFixed(5)}° ${value >= 0 ? positive : negative}`;
  }

  function showVessel(vessel) {
    if (!vessel) {
      vesselFields.data.hidden = true;
      vesselFields.empty.hidden = false;
      vesselFields.empty.textContent = selectedMmsi
        ? '이 시각에는 선택한 선박의 최근 AIS 위치가 없습니다. 시간을 바꿔 보세요.'
        : '지도에서 선박을 클릭하면 항적과 정보를 볼 수 있습니다.';
      vesselFields.heading.textContent = selectedTrack?.vessel_name || '선박 정보';
      return;
    }
    vesselFields.heading.textContent = vessel.vessel_name;
    vesselFields.empty.hidden = true;
    vesselFields.data.hidden = false;
    vesselFields.vesselClass.textContent = vessel.unified_label;
    vesselFields.mmsi.textContent = vessel.mmsi;
    vesselFields.sog.textContent = vessel.sog_kn === null ? '자료 없음' : `${vessel.sog_kn.toFixed(1)} kn`;
    vesselFields.cog.textContent = vessel.cog_deg === null ? '자료 없음' : `${vessel.cog_deg.toFixed(1)}°`;
    vesselFields.lat.textContent = formatCoordinate(vessel.lat, 'N', 'S');
    vesselFields.lon.textContent = formatCoordinate(vessel.lon, 'E', 'W');
    vesselFields.time.textContent = `${vessel.timestamp_utc.slice(11, 19)} UTC`;
    vesselFields.note.textContent = selectedTrack
      ? `하루 항적 ${selectedTrack.points.length.toLocaleString('ko-KR')}개 위치 기록 · 선택 시각의 위치는 최대 ${metadata.position_max_age_minutes}분 전 기록입니다.`
      : `선택 시각의 위치는 최대 ${metadata.position_max_age_minutes}분 전 기록입니다.`;
  }

  function renderSelectedTrack() {
    selectedLayer.clearLayers();
    if (!selectedTrack || !metadata) return;
    const limit = Date.parse(selectedIsoTime());
    const color = colors[selectedTrack.unified_label] || colors.Other;
    let segment = [];
    let previousTime = null;
    const addSegment = () => {
      if (segment.length > 1) {
        L.polyline(segment, { color, weight: 4, opacity: 0.88, interactive: false }).addTo(selectedLayer);
      }
    };
    for (const point of selectedTrack.points) {
      const at = Date.parse(point.timestamp_utc);
      if (at > limit) break;
      if (previousTime !== null && at - previousTime > 30 * 60000) {
        addSegment();
        segment = [];
      }
      segment.push([point.lat, point.lon]);
      previousTime = at;
    }
    addSegment();
  }

  async function selectVessel(vessel) {
    selectedMmsi = vessel.mmsi;
    selectedTrack = null;
    showVessel(vessel);
    map.panTo([vessel.lat, vessel.lon]);
    renderMarkers();
    try {
      const url = root.dataset.trackUrlTemplate.replace('MMSI_PLACEHOLDER', encodeURIComponent(vessel.mmsi));
      const track = await getJson(url);
      if (selectedMmsi !== vessel.mmsi) return;
      selectedTrack = track;
      renderSelectedTrack();
      showVessel(currentPositions.get(selectedMmsi));
    } catch (error) {
      if (selectedMmsi === vessel.mmsi) setStatus(`항적을 불러오지 못했습니다: ${error.message}`, true);
    }
  }

  function renderMarkers() {
    markersLayer.clearLayers();
    trailsLayer.clearLayers();
    for (const vessel of currentPositions.values()) {
      const color = colors[vessel.unified_label] || colors.Other;
      if (vessel.trail.length > 1 && vessel.mmsi !== selectedMmsi) {
        L.polyline(vessel.trail, { color, weight: 2, opacity: 0.58, interactive: false })
          .addTo(trailsLayer);
      }
      const selected = vessel.mmsi === selectedMmsi;
      L.circleMarker([vessel.lat, vessel.lon], {
        radius: selected ? 8 : 5,
        color: selected ? '#243840' : '#ffffff', weight: selected ? 2 : 1,
        fillColor: color, fillOpacity: 0.95,
      }).on('click', () => selectVessel(vessel)).addTo(markersLayer);
    }
    renderSelectedTrack();
  }

  async function loadScene() {
    if (!metadata) return;
    if (pendingRequest) pendingRequest.abort();
    const request = new AbortController();
    pendingRequest = request;
    const sequence = ++sceneSequence;
    const url = new URL(root.dataset.positionsUrl, window.location.href);
    url.searchParams.set('at', selectedIsoTime());
    if (controls.filter.value) url.searchParams.set('class', controls.filter.value);
    try {
      const scene = await getJson(url, request.signal);
      if (sequence !== sceneSequence) return;
      currentPositions = new Map(scene.vessels.map(vessel => [vessel.mmsi, vessel]));
      renderMarkers();
      showVessel(currentPositions.get(selectedMmsi));
      controls.count.textContent = `${scene.count.toLocaleString('ko-KR')}척 표시`;
      setStatus(`선택 시각 ${scene.at_utc.slice(11, 16)} UTC · 최근 ${metadata.position_max_age_minutes}분 이내 AIS 위치를 표시합니다.`);
    } catch (error) {
      if (error.name === 'AbortError' || sequence !== sceneSequence) return;
      currentPositions.clear();
      renderMarkers();
      showVessel(null);
      controls.count.textContent = '위치 조회 실패';
      setStatus(error.message, true);
    } finally {
      if (pendingRequest === request) pendingRequest = null;
    }
  }

  function scheduleScene(immediate = false) {
    clearTimeout(debounceTimer);
    updateClock();
    if (immediate) loadScene();
    else debounceTimer = setTimeout(loadScene, 120);
  }

  function stopPlayback() {
    if (playbackTimer !== null) clearInterval(playbackTimer);
    playbackTimer = null;
    controls.play.textContent = '▶ 재생';
    controls.play.setAttribute('aria-label', '시간 재생');
  }

  function togglePlayback() {
    if (playbackTimer !== null) {
      stopPlayback();
      return;
    }
    if (Number(controls.slider.value) >= Number(controls.slider.max)) controls.slider.value = 0;
    controls.play.textContent = '❚❚ 일시정지';
    controls.play.setAttribute('aria-label', '시간 재생 일시정지');
    scheduleScene(true);
    playbackTimer = setInterval(() => {
      const next = Math.min(Number(controls.slider.max),
        Number(controls.slider.value) + Number(controls.speed.value) * 2);
      controls.slider.value = String(next);
      scheduleScene(true);
      if (next >= Number(controls.slider.max)) stopPlayback();
    }, 400);
  }

  async function initialize() {
    if (!window.L) {
      setStatus('지도 라이브러리를 불러오지 못했습니다. 인터넷 연결을 확인하세요.', true);
      return;
    }
    map = L.map('traffic-map', { preferCanvas: true }).setView([37.80, -122.35], 10);
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 18,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(map);
    L.control.scale({ imperial: false }).addTo(map);
    trailsLayer = L.layerGroup().addTo(map);
    selectedLayer = L.layerGroup().addTo(map);
    markersLayer = L.layerGroup().addTo(map);

    try {
      metadata = await getJson(root.dataset.metaUrl);
      controls.date.textContent = `${metadata.date_utc} · UTC · ${metadata.vessel_count.toLocaleString('ko-KR')}척`;
      controls.slider.max = String(Math.floor((Date.parse(metadata.end_utc) - Date.parse(metadata.start_utc)) / 60000));
      controls.slider.value = String(Math.min(720, Number(controls.slider.max)));
      for (const item of metadata.classes) {
        const option = document.createElement('option');
        option.value = item.name;
        option.textContent = `${item.name} (${item.vessels}척)`;
        controls.filter.appendChild(option);
      }
      controls.fit.disabled = false;
      controls.filter.disabled = false;
      controls.slider.disabled = false;
      controls.play.disabled = false;
      controls.speed.disabled = false;
      map.fitBounds(metadata.bounds, { padding: [20, 20] });
      updateClock();
      await loadScene();
    } catch (error) {
      setStatus(error.message, true);
      controls.count.textContent = 'AIS 데이터 없음';
    }
  }

  controls.fit.addEventListener('click', () => metadata && map.fitBounds(metadata.bounds, { padding: [20, 20] }));
  controls.filter.addEventListener('change', () => {
    if (selectedTrack && controls.filter.value && selectedTrack.unified_label !== controls.filter.value) {
      selectedMmsi = '';
      selectedTrack = null;
      selectedLayer.clearLayers();
    }
    scheduleScene(true);
  });
  controls.slider.addEventListener('input', () => scheduleScene());
  controls.slider.addEventListener('change', () => scheduleScene(true));
  controls.play.addEventListener('click', togglePlayback);
  initialize();
})();
