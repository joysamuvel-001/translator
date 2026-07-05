import { Mic, Square, RotateCcw, Plus, ChevronDown } from 'lucide-react'

const LANGUAGES = [
  { code: 'as', label: 'Assamese' },
  { code: 'bn', label: 'Bengali' },
  { code: 'brx', label: 'Bodo' },
  { code: 'doi', label: 'Dogri' },
  { code: 'gu', label: 'Gujarati' },
  { code: 'hi', label: 'Hindi' },
  { code: 'kn', label: 'Kannada' },
  { code: 'kok', label: 'Konkani' },
  { code: 'ks', label: 'Kashmiri' },
  { code: 'mai', label: 'Maithili' },
  { code: 'ml', label: 'Malayalam' },
  { code: 'mni', label: 'Manipuri' },
  { code: 'mr', label: 'Marathi' },
  { code: 'ne', label: 'Nepali' },
  { code: 'or', label: 'Odia' },
  { code: 'pa', label: 'Punjabi' },
  { code: 'sa', label: 'Sanskrit' },
  { code: 'sat', label: 'Santali' },
  { code: 'sd', label: 'Sindhi' },
  { code: 'ta', label: 'Tamil' },
  { code: 'te', label: 'Telugu' },
  { code: 'ur', label: 'Urdu' },
]

export default function RecordingPanel({
  isRecording,
  elapsed,
  busy,
  onToggle,
  onRetryLast,
  onNewSession,
  canRetry,
  language,
  onLanguageChange,
}) {
  return (
    <section className="px-6 py-6 border-b border-ink-border">
      <p className="text-[0.7rem] font-mono uppercase tracking-[0.14em] text-text-faint mb-4">
        Recording
      </p>

      <div className="mb-5">
        <label className="block text-[0.7rem] text-text-faint mb-1.5">
          Speaker language
        </label>
        <div className="relative">
          <select
            value={language}
            onChange={(e) => onLanguageChange(e.target.value)}
            disabled={isRecording || busy}
            className="w-full appearance-none rounded-lg border border-ink-border bg-ink-raised
                       py-2 pl-3 pr-8 text-[0.82rem] text-text-primary
                       focus:outline-none focus:border-vital-teal/50 disabled:opacity-50
                       cursor-pointer"
          >
            {LANGUAGES.map((l) => (
              <option key={l.code} value={l.code}>
                {l.label}
              </option>
            ))}
          </select>
          <ChevronDown className="w-3.5 h-3.5 text-text-faint absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
        </div>
      </div>

      <div className="flex flex-col items-center">
        <button
          onClick={onToggle}
          disabled={busy}
          className={`relative w-20 h-20 rounded-full flex items-center justify-center border transition-colors
            ${isRecording
              ? 'bg-vital-coral/15 border-vital-coral/50'
              : 'bg-ink-raised border-ink-border hover:border-vital-teal/40'}`}
        >
          {isRecording && (
            <span className="absolute inset-0 rounded-full bg-vital-coral/20 vitals-pulse" />
          )}
          {isRecording ? (
            <Square className="w-6 h-6 text-vital-coral relative" />
          ) : (
            <Mic className="w-7 h-7 text-text-muted relative" />
          )}
        </button>

        <div className="mt-3 h-7 flex items-center gap-[3px]">
          {isRecording ? (
            <VitalsBars />
          ) : (
            <span className="text-[0.78rem] text-text-faint">
              {busy ? 'processing…' : 'Tap to record'}
            </span>
          )}
        </div>
        {isRecording && (
          <span className="font-mono text-[0.78rem] text-vital-coral mt-1">
            {elapsed.toFixed(1)}s
          </span>
        )}
      </div>

      <div className="mt-5 grid grid-cols-2 gap-2">
        <button
          onClick={onRetryLast}
          disabled={!canRetry || busy}
          className="flex items-center justify-center gap-1.5 rounded-lg border border-ink-border
                     bg-ink-raised py-2 text-[0.78rem] text-text-muted hover:text-text-primary
                     hover:border-text-faint/40 transition-colors disabled:opacity-40"
        >
          <RotateCcw className="w-3.5 h-3.5" /> Retry last
        </button>
        <button
          onClick={onNewSession}
          className="flex items-center justify-center gap-1.5 rounded-lg border border-vital-indigo/30
                     bg-vital-indigo/10 py-2 text-[0.78rem] text-vital-indigo hover:bg-vital-indigo/15
                     transition-colors"
        >
          <Plus className="w-3.5 h-3.5" /> New session
        </button>
      </div>
    </section>
  )
}

function VitalsBars() {
  const heights = [6, 14, 22, 12, 18, 8, 16]
  return (
    <div className="flex items-end gap-[3px] h-6">
      {heights.map((h, i) => (
        <span
          key={i}
          className="w-[3px] rounded-full bg-vital-coral vitals-pulse"
          style={{ height: `${h}px`, animationDelay: `${i * 0.09}s` }}
        />
      ))}
    </div>
  )
}