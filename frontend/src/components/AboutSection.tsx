export function AboutSection() {
  return (
    <section className="about" id="about" aria-labelledby="about-h">
      <div className="section-head" data-reveal>
        <p className="eyebrow">06 · About</p>
        <h2 id="about-h">What this is, and what it isn’t</h2>
      </div>
      <div className="about-grid" data-reveal-stagger>
        <article className="glass" data-tilt>
          <h3>How it works</h3>
          <ol>
            <li><b>Voice.</b> 22 measures of a sustained vowel; the 4 most informative are kept.</li>
            <li><b>Spiral.</b> The image becomes edge-direction features (HOG), compressed to 4 numbers.</li>
            <li><b>Qubits.</b> The 8 numbers become rotation angles on 8 simulated qubits, then trainable layers.</li>
            <li><b>Reasons.</b> SHAP splits the score into each feature’s push, the same way for every model.</li>
          </ol>
        </article>
        <article className="glass" data-tilt>
          <h3>Honest limits</h3>
          <ul>
            <li>Simulated quantum circuits on a normal computer. No quantum advantage is claimed.</li>
            <li>Small data: 195 voice recordings with 32 subject IDs (8 healthy) and 102 spirals.</li>
            <li>Voice and spirals come from different people and were paired by their known labels for the experiment.</li>
            <li>Research only: not a medical device, and clinical use is outside the study’s scope. <a href="#research">See the results and limits</a>.</li>
          </ul>
        </article>
        <article className="glass" data-tilt>
          <h3>Data and credits</h3>
          <ul>
            <li>Voice: UCI Parkinsons dataset, Little et al. (2007), CC BY 4.0.</li>
            <li>Spirals: Kaggle “Parkinson’s Drawings” (Zham et al., 2017).</li>
            <li>Built with PennyLane, scikit-learn, SHAP, FastAPI, React and Three.js.</li>
          </ul>
          <h3 className="team-h">Team</h3>
          <ul className="team">
            <li><b>Shreyas</b> data, classical and quantum models</li>
            <li><b>Thanusri V</b> web app</li>
            <li><b>Nayeem</b> design</li>
            <li><b>Altaf</b> documentation</li>
          </ul>
        </article>
      </div>
    </section>
  );
}
