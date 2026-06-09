import { useState, useRef } from "react"
import { predictBatch } from "../api"

export default function BatchTab({ onPredictionDone }) {
  const [files, setFiles] = useState([])
  const [previews, setPreviews] = useState([])
  const [results, setResults] = useState(null)
  const [predicting, setPredicting] = useState(false)
  const fileRef = useRef(null)

  const handleFiles = (newFiles) => {
    const arr = Array.from(newFiles).filter((f) => f.type.startsWith("image/"))
    setFiles(arr)
    setPreviews(arr.map((f) => URL.createObjectURL(f)))
    setResults(null)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    handleFiles(e.dataTransfer.files)
  }

  const handlePredict = async () => {
    if (files.length === 0) return
    setPredicting(true)
    try {
      const data = await predictBatch(files)
      setResults(data.results)
      onPredictionDone()
    } catch (err) {
      setResults([{ filename: "错误", error: err.message }])
    }
    setPredicting(false)
  }

  const downloadCSV = () => {
    if (!results) return
    const headers = ["文件名", "Top-1 类别", "置信度", "Top-2 类别", "置信度", "Top-3 类别", "置信度"]
    const rows = results.map((r) => {
      if (r.error) return [r.filename, `错误: ${r.error}`, "", "", "", "", ""]
      const row = [r.filename]
      for (let i = 0; i < 3; i++) {
        const p = r.results[i]
        row.push(p ? p["class"] : "")
        row.push(p ? p.confidence.toFixed(4) : "")
      }
      return row
    })
    const csv = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n")
    const blob = new Blob(["\ufeff" + csv], { type: "text/csv;charset=utf-8" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "batch_results.csv"
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <>
      <div className="card">
        <h3>上传多张图片</h3>
        <div className="row">
          <div className="col" style={{ flex: 2, minWidth: 260 }}>
            <div
              className="upload-area"
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileRef.current?.click()}
              style={{ cursor: "pointer" }}
            >
              <div className="upload-icon">📁</div>
              <div className="upload-text">拖拽图片到此处，或点击选择</div>
              <div className="upload-hint">支持 JPG / PNG 格式，可一次上传多张</div>
            </div>
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              multiple
              style={{ display: "none" }}
              onChange={(e) => handleFiles(e.target.files)}
            />
            <button
              className="btn btn-primary"
              disabled={files.length === 0 || predicting}
              onClick={handlePredict}
              style={{ marginTop: 12, width: "100%" }}
            >
              {predicting ? <span className="loading" /> : "开始识别"}
            </button>
          </div>

          <div className="col" style={{ flex: 5, minWidth: 400 }}>
            {previews.length > 0 && (
              <>
                <div style={{ fontSize: 13, color: "var(--olive-gray)", marginBottom: 4 }}>
                  已上传（{previews.length} 张）
                </div>
                <div className="gallery">
                  {previews.map((url, i) => (
                    <div key={i} className="gallery-item">
                      <img src={url} alt="" />
                      <span className="gallery-num">{i + 1}</span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {results && (
        <div className="card" style={{ marginTop: 14 }}>
          <h3>识别结果</h3>
          <button className="btn btn-secondary btn-sm" onClick={downloadCSV} style={{ marginBottom: 8 }}>
            下载 CSV
          </button>
          <div style={{ overflowX: "auto" }}>
            <table className="batch-table">
              <thead>
                <tr>
                  <th>文件名</th>
                  <th>Top-1 类别</th>
                  <th>置信度</th>
                  <th>Top-2 类别</th>
                  <th>置信度</th>
                  <th>Top-3 类别</th>
                  <th>置信度</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => {
                  if (r.error) {
                    return (
                      <tr key={i}>
                        <td>{r.filename}</td>
                        <td colSpan={6} style={{ color: "var(--terracotta)" }}>错误: {r.error}</td>
                      </tr>
                    )
                  }
                  return (
                    <tr key={i}>
                      <td>{r.filename}</td>
                      {[0, 1, 2].map((j) => (
                        <td key={`c${j}`}>
                          {r.results[j] ? r.results[j]["class"] : ""}
                        </td>
                      ))}
                      {[0, 1, 2].map((j) => (
                        <td key={`p${j}`}>
                          {r.results[j] ? r.results[j].confidence.toFixed(4) : ""}
                        </td>
                      ))}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  )
}
