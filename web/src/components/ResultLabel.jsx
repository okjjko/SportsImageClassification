export default function ResultLabel({ results }) {
  if (!results || results.length === 0) {
    return <div className="result-empty">无结果</div>
  }

  if (results[0]["class"] && results[0]["class"].startsWith("错误")) {
    return (
      <div className="result-empty" style={{ color: "var(--terracotta)" }}>
        {results[0]["class"]}
      </div>
    )
  }

  const maxConf = Math.max(...results.map((r) => r.confidence))

  return (
    <div className="result-label">
      {results.map((r, i) => (
        <div key={i} className="result-item">
          <span className="rank">#{i + 1}</span>
          <span className="label">{r["class"]}</span>
          <div className="bar-wrap">
            <div
              className="bar-fill"
              style={{ width: `${maxConf > 0 ? (r.confidence / maxConf) * 100 : 0}%` }}
            />
          </div>
          <span className="prob">{(r.confidence * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  )
}
