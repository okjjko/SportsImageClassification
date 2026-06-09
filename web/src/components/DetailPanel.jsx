import ResultLabel from "./ResultLabel"

export default function DetailPanel({ entry, onBack }) {
  if (!entry) return null

  const imgUrl = `/api/files/history/${entry.filepath.split("/").pop().split("\\").pop()}`

  return (
    <div className="detail-panel card">
      <button className="btn btn-secondary btn-sm" onClick={onBack}>
        ← 返回识别
      </button>

      <div className="detail-top" style={{ marginTop: 16 }}>
        <div className="detail-image-wrap">
          <img src={imgUrl} alt={entry.filename} />
        </div>

        <div style={{ flex: 1, minWidth: 0 }}>
          <ResultLabel results={entry.results} />
          <div className="detail-meta">
            <div>📄 {entry.filename}</div>
            <div>🕐 {entry.timestamp}</div>
            <div>📁 来源: {entry.source}</div>
            <div>🔗 {entry.filepath}</div>
          </div>
        </div>
      </div>
    </div>
  )
}
