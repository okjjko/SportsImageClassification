import { useEffect, useState, useRef } from "react"
import { getExamples, predictSingle } from "../api"
import ResultLabel from "./ResultLabel"

export default function SingleTab({ onPredictionDone }) {
  const [examples, setExamples] = useState([])
  const [preview, setPreview] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [predicting, setPredicting] = useState(false)
  const [result, setResult] = useState(null)
  const fileRef = useRef(null)

  useEffect(() => {
    getExamples().then((data) => setExamples(data.examples || []))
  }, [])

  const handleFile = (file) => {
    if (!file) return
    setPreview(file)
    setPreviewUrl(URL.createObjectURL(file))
    setResult(null)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith("image/")) handleFile(file)
  }

  const handlePredict = async () => {
    if (!preview) return
    setPredicting(true)
    try {
      const data = await predictSingle(preview)
      setResult(data.results)
      onPredictionDone()
    } catch (err) {
      setResult([{ class: `错误: ${err.message}`, confidence: 1 }])
    }
    setPredicting(false)
  }

  const handleClear = () => {
    setPreview(null)
    setPreviewUrl(null)
    setResult(null)
    if (fileRef.current) fileRef.current.value = ""
  }

  const handleExampleClick = async (url) => {
    try {
      const resp = await fetch(url)
      const blob = await resp.blob()
      const file = new File([blob], url.split("/").pop(), { type: blob.type })
      handleFile(file)
    } catch {
      // ignore
    }
  }

  return (
    <div className="row">
      <div className="col col-4">
        <div className="card">
          <h3>上传图片</h3>

          <div
            className="upload-area"
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileRef.current?.click()}
          >
            {previewUrl ? (
              <img src={previewUrl} alt="preview" className="preview-img" style={{ maxHeight: "220px" }} />
            ) : (
              <>
                <div className="upload-icon">🖼️</div>
                <div className="upload-text">拖拽图片到此处，或点击选择</div>
                <div className="upload-hint">支持 JPG / PNG 格式</div>
              </>
            )}
          </div>
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={(e) => handleFile(e.target.files[0])}
          />

          {preview && (
            <div className="btn-group">
              <button className="btn btn-primary" disabled={predicting} onClick={handlePredict}>
                {predicting ? <span className="loading" /> : "开始识别"}
              </button>
              <button className="btn btn-secondary" onClick={handleClear}>
                清除
              </button>
            </div>
          )}

          {examples.length > 0 && (
            <>
              <div className="example-label" style={{ marginTop: 16 }}>
                快速测试 — 点击下方示例图片
              </div>
              <div className="example-grid">
                {examples.map((ex) => (
                  <button key={ex.name} className="example-thumb" onClick={() => handleExampleClick(ex.url)}>
                    <img src={ex.url} alt={ex.name} />
                    <span>{ex.name}</span>
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      </div>

      <div className="col col-5">
        <div className="card">
          <h3>识别结果</h3>
          {result ? <ResultLabel results={result} /> : <div className="result-empty">等待识别...</div>}
        </div>
      </div>
    </div>
  )
}
