export function SignalsSection() {
  return (
    <section className="signals" id="signals" data-anchor="1" aria-labelledby="signals-h">
      <div className="section-head" data-reveal>
        <p className="eyebrow">01 · Two signals</p>
        <h2 id="signals-h">The same symptom shows up twice</h2>
        <p className="lede">
          Tremor and stiffness affect the hand and the voice. Each 3D pair shows a spiral and a sustained “aaah”.
        </p>
      </div>
      <div className="signal-cards" data-reveal-stagger>
        <article className="glass signal-card healthy" data-tilt>
          <h3>Steady</h3>
          <p>Even loops on paper. A voice wave that repeats with the same timing and height every cycle.</p>
          <dl>
            <div><dt>Jitter</dt><dd>low: pitch holds steady</dd></div>
            <div><dt>Shimmer</dt><dd>low: loudness holds steady</dd></div>
          </dl>
        </article>
        <article className="glass signal-card tremor" data-tilt>
          <h3>With tremor</h3>
          <p>Shaky, uneven loops. A voice wave whose timing (jitter) and height (shimmer) wobble cycle to cycle.</p>
          <dl>
            <div><dt>Jitter</dt><dd>higher: pitch wobbles</dd></div>
            <div><dt>Shimmer</dt><dd>higher: loudness wobbles</dd></div>
          </dl>
        </article>
      </div>
      <p className="fine">3D shapes are illustrations, not patient data.</p>
    </section>
  );
}
