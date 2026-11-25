'use strict';
// Pure helpers shared by the canvas renderer and tests.
globalThis.SeaSound = (() => {
  const axis = a => Array.isArray(a) && a.length && a.every((v,i) => Number.isFinite(v) && (!i || v >= a[i-1]));
  function normalize(raw) {
    const f=raw.frequencies ?? raw.freqs ?? raw.f, t=raw.times ?? raw.t;
    const z=raw.db ?? raw.S_db ?? raw.stft_db ?? raw.Zxx_db;
    if (!axis(f) || !axis(t) || f[0]<0 || t[0]<0 || !Array.isArray(z) || z.length!==f.length ||
        !z.every(row=>Array.isArray(row) && row.length===t.length && row.every(Number.isFinite))) {
      throw new Error('스펙트로그램 응답 형식 오류: frequencies, times, db[주파수][시간]을 확인하세요.');
    }
    const duration = Number.isFinite(raw.duration) && raw.duration > 0 ? raw.duration : null;
    return {f,t,z,duration};
  }
  function ranges(values) {
    const parsed=values.map(v=>String(v).trim()==='' ? NaN : Number(v));
    const [lo,hi,dlo,dhi]=parsed;
    if (!parsed.every(Number.isFinite) || lo<0 || lo>=hi || dlo>=dhi) throw new Error('주파수·dB 하한은 상한보다 작아야 하며 주파수는 0 이상이어야 합니다.');
    return {lo,hi,dlo,dhi};
  }
  const clamp=v=>Math.max(0,Math.min(1,v));
  const seek=(fraction,duration)=>clamp(fraction)*duration;
  function nearest(arr,v) {
    let l=0,r=arr.length-1;
    while(l<r){const m=Math.floor((l+r)/2);if(arr[m]<v)l=m+1;else r=m;}
    return l>0 && Math.abs(arr[l-1]-v)<Math.abs(arr[l]-v)?l-1:l;
  }
  function palette(v) {
    v=clamp(v);const a=v<.5?[15,25,70]:[35,160,160], b=v<.5?[35,160,160]:[250,230,90];
    const w=v<.5?v*2:(v-.5)*2;return a.map((n,i)=>Math.round(n+(b[i]-n)*w));
  }
  // Slaney scale, matching librosa's default mel filter bank.
  const logStep = Math.log(6.4) / 27;
  const toMel = hz => hz < 1000 ? hz / (200 / 3) : 15 + Math.log(hz / 1000) / logStep;
  const fromMel = mel => mel < 15 ? mel * (200 / 3) : 1000 * Math.exp((mel - 15) * logStep);
  return {normalize,ranges,clamp,seek,nearest,palette,toMel,fromMel};
})();
