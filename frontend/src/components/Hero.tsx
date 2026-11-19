export function Hero() {
  return (
    <section className="hero" id="top" data-anchor="0">
      <div className="hero-copy">
        <p className="eyebrow">Research prototype · voice + spiral + 8 qubits</p>
        <h1>
          A spiral, a voice,
          <br />
          and eight qubits.
        </h1>
        <p className="lede">
          Parkinson’s often shows in a shaky hand-drawn spiral and in small wobbles of the voice. This prototype reads
          both, scores them with a simulated quantum classifier, and shows exactly which signals moved the score.
        </p>
        <div className="hero-actions">
          <a className="btn btn-primary" href="#screen">Try the screening</a>
          <a className="btn btn-ghost" href="#signals">See how it works</a>
        </div>
        <ul className="hero-facts" aria-label="Data behind the models">
          <li><b>195</b> voice recordings, 32 subject IDs</li>
          <li><b>102</b> hand-drawn spirals</li>
          <li><b>8</b> simulated qubits</li>
        </ul>
        <p className="hero-note">Not a medical device and not a diagnosis.</p>
      </div>
      <a className="scroll-cue" href="#signals" aria-label="Scroll to the next section">
        <span />
      </a>
    </section>
  );
}
