import { useCallback, useRef, useState } from 'react'

const TARGET_SR = 16000

function encodeWav(samples) {
  const int16 = new Int16Array(samples.length)
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff
  }
  const dataLen = int16.buffer.byteLength
  const buf = new ArrayBuffer(44 + dataLen)
  const view = new DataView(buf)
  const w = (off, val, size) => {
    if (size === 1) view.setUint8(off, val)
    else if (size === 2) view.setUint16(off, val, true)
    else view.setUint32(off, val, true)
  }
  const ws = (off, str) => { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)) }
  ws(0, 'RIFF'); w(4, 36 + dataLen, 4); ws(8, 'WAVE'); ws(12, 'fmt ')
  w(16, 16, 4); w(20, 1, 2); w(22, 1, 2); w(24, TARGET_SR, 4)
  w(28, TARGET_SR * 2, 4); w(32, 2, 2); w(34, 16, 2)
  ws(36, 'data'); w(40, dataLen, 4)
  new Uint8Array(buf, 44).set(new Uint8Array(int16.buffer))
  return new Blob([buf], { type: 'audio/wav' })
}

export function useRecorder() {
  const [isRecording, setIsRecording] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const ctxRef = useRef(null)
  const processorRef = useRef(null)
  const sourceRef = useRef(null)
  const samplesRef = useRef([])
  const timerRef = useRef(null)
  const startedAtRef = useRef(0)

  const start = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    const ctx = new AudioContext({ sampleRate: TARGET_SR })
    ctxRef.current = ctx
    const source = ctx.createMediaStreamSource(stream)
    sourceRef.current = source
    const processor = ctx.createScriptProcessor(4096, 1, 1)
    processorRef.current = processor
    samplesRef.current = []
    processor.onaudioprocess = (e) => {
      samplesRef.current.push(new Float32Array(e.inputBuffer.getChannelData(0)))
    }
    source.connect(processor)
    processor.connect(ctx.destination)
    startedAtRef.current = Date.now()
    setIsRecording(true)
    setElapsed(0)
    timerRef.current = setInterval(() => {
      setElapsed((Date.now() - startedAtRef.current) / 1000)
    }, 100)
  }, [])

  const stop = useCallback(() => {
    return new Promise((resolve) => {
      clearInterval(timerRef.current)
      setIsRecording(false)
      const source = sourceRef.current
      const processor = processorRef.current
      const ctx = ctxRef.current
      if (source) source.disconnect()
      if (processor) processor.disconnect()
      const all = samplesRef.current
      const total = all.reduce((n, c) => n + c.length, 0)
      const merged = new Float32Array(total)
      let offset = 0
      for (const chunk of all) { merged.set(chunk, offset); offset += chunk.length }
      if (ctx) ctx.close().catch(() => {})
      resolve(encodeWav(merged))
    })
  }, [])

  return { isRecording, elapsed, start, stop }
}