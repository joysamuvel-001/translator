const langColor = {
  Hindi: 'text-vital-amber border-vital-amber/30 bg-vital-amber/10',
  English: 'text-vital-indigo border-vital-indigo/30 bg-vital-indigo/10',
  Tamil: 'text-vital-teal border-vital-teal/30 bg-vital-teal/10',
  Telugu: 'text-vital-coral border-vital-coral/30 bg-vital-coral/10',
  Kannada: 'text-vital-teal border-vital-teal/30 bg-vital-teal/10',
  Malayalam: 'text-vital-coral border-vital-coral/30 bg-vital-coral/10',
  Bengali: 'text-vital-amber border-vital-amber/30 bg-vital-amber/10',
  Marathi: 'text-vital-indigo border-vital-indigo/30 bg-vital-indigo/10',
  Gujarati: 'text-vital-teal border-vital-teal/30 bg-vital-teal/10',
  Punjabi: 'text-vital-coral border-vital-coral/30 bg-vital-coral/10',
  Odia: 'text-vital-amber border-vital-amber/30 bg-vital-amber/10',
  Urdu: 'text-vital-indigo border-vital-indigo/30 bg-vital-indigo/10',
  Assamese: 'text-vital-teal border-vital-teal/30 bg-vital-teal/10',
  Nepali: 'text-vital-coral border-vital-coral/30 bg-vital-coral/10',
  Sanskrit: 'text-vital-amber border-vital-amber/30 bg-vital-amber/10',
}
const defaultLangStyle = 'text-text-muted border-ink-border bg-ink-raised'

export default function TranscriptCard({ turn }) {
  const matched = turn.match_confidence != null
  const initial = turn.speaker_name.trim().charAt(0).toUpperCase()

  return (
    <article className="group rounded-xl2 border border-ink-border bg-ink-surface/60 hover:bg-ink-surface
                         transition-colors px-5 py-4 shadow-panel">
      <header className="flex items-center gap-3">
        <div
          className={`w-8 h-8 rounded-full flex items-center justify-center text-[0.78rem] font-medium border
            ${matched ? 'bg-vital-teal/15 border-vital-teal/30 text-vital-teal' : 'bg-ink-raised border-ink-border text-text-faint'}`}
        >
          {initial}
        </div>
        <div className="flex flex-col leading-tight">
          <span className="text-[0.92rem] font-medium text-text-primary">{turn.speaker_name}</span>
          <span className="text-[0.68rem] font-mono text-text-faint">
            {turn.start_sec.toFixed(1)}s – {turn.end_sec.toFixed(1)}s
          </span>
        </div>

        <div className="ml-auto flex items-center gap-2">
          <span
            className={`text-[0.68rem] font-mono px-2 py-0.5 rounded-full border ${langColor[turn.detected_language] || defaultLangStyle}`}
          >
            {turn.detected_language}
          </span>
          {matched ? (
            <span className="text-[0.68rem] font-mono px-2 py-0.5 rounded-full border border-vital-teal/30 bg-vital-teal/10 text-vital-teal">
              {Math.round(turn.match_confidence * 100)}% match
            </span>
          ) : (
            <span className="text-[0.68rem] font-mono px-2 py-0.5 rounded-full border border-vital-coral/30 bg-vital-coral/10 text-vital-coral">
              unverified
            </span>
          )}
        </div>
      </header>

      <p className="mt-3 text-[0.95rem] leading-relaxed text-text-primary">{turn.source_text}</p>

      {turn.detected_language !== 'English' && (
        <p className="mt-2 pt-2 border-t border-ink-border/70 text-[0.84rem] leading-relaxed text-text-muted italic">
          {turn.translated_text}
        </p>
      )}

      <footer className="mt-2.5 text-[0.68rem] font-mono text-text-faint">
        via {turn.speaker_id ? `speaker_${turn.speaker_id.slice(0, 6)}` : 'unidentified_voice'}
      </footer>
    </article>
  )
}