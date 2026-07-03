import { useState } from 'react'
import { Mic, Square, Check } from 'lucide-react'
import { useRecorder } from '../hooks/useRecorder'
import { api } from '../lib/api'

export default function EnrollmentPanel({ onEnrolled }) {
  const [name, setName] = useState('')
  const [status, setStatus] = useState(null) // null | 'saving' | 'done' | error string
  const { isRecording, elapsed, start, stop } = useRecorder()

  const handleToggle = async () => {
    if (!isRecording) {
      if (!name.trim()) {
        setStatus('Enter a name before recording.')
        return
      }
      setStatus(null)
      await start()
      return
    }
    const blob = await stop()
    setStatus('saving')
    try {
      const speaker = await api.enrollSpeaker(name.trim(), blob)
      setStatus('done')
      setName('')
      onEnrolled?.(speaker)
      setTimeout(() => setStatus(null), 1800)
    } catch (err) {
      setStatus(err.message)
    }
  }

  const tooShort = elapsed > 0 && elapsed < 5
  const ready = elapsed >= 5

  return (
    <section className="px-6 py-6 border-b border-ink-border">
      <p className="text-[0.7rem] font-mono uppercase tracking-[0.14em] text-text-faint mb-3">
        Speaker enrollment
      </p>
      <p className="text-[0.82rem] leading-relaxed text-text-muted mb-4">
        Record a 5–10s sample per person so TitaNet can recognize their voice
        in every future session.
      </p>

      <input
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="Person's name (e.g. Dr. Priya)"
        disabled={isRecording}
        className="w-full mb-3 bg-ink-raised border border-ink-border rounded-lg px-3.5 py-2.5
                   text-[0.88rem] text-text-primary placeholder:text-text-faint
                   focus:outline-none focus:border-vital-teal/50 focus:ring-1 focus:ring-vital-teal/30
                   transition-colors disabled:opacity-50"
      />

      <button
        onClick={handleToggle}
        disabled={status === 'saving'}
        className={`w-full flex items-center justify-center gap-2 rounded-lg py-2.5 text-[0.86rem] font-medium
          transition-colors border
          ${isRecording
            ? 'bg-vital-coral/15 border-vital-coral/40 text-vital-coral'
            : 'bg-vital-teal/10 border-vital-teal/35 text-vital-teal hover:bg-vital-teal/15'}`}
      >
        {isRecording ? <Square className="w-3.5 h-3.5" /> : <Mic className="w-4 h-4" />}
        {isRecording ? `Stop · ${elapsed.toFixed(1)}s` : 'Record sample'}
      </button>

      <div className="mt-2.5 h-4 flex items-center justify-center">
        {isRecording && tooShort && (
          <span className="text-[0.72rem] font-mono text-vital-amber">
            keep going — {(5 - elapsed).toFixed(1)}s to minimum
          </span>
        )}
        {isRecording && ready && (
          <span className="text-[0.72rem] font-mono text-vital-teal">sample length is good</span>
        )}
        {status === 'saving' && (
          <span className="text-[0.72rem] font-mono text-text-muted">enrolling voiceprint…</span>
        )}
        {status === 'done' && (
          <span className="text-[0.72rem] font-mono text-vital-teal flex items-center gap-1">
            <Check className="w-3 h-3" /> enrolled
          </span>
        )}
        {status && !['saving', 'done'].includes(status) && (
          <span className="text-[0.72rem] font-mono text-vital-coral">{status}</span>
        )}
      </div>
    </section>
  )
}
