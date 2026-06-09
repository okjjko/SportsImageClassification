import { useEffect, useState } from "react"
import { getHistory, clearHistory, getHistoryDetail } from "../api"

export default function Sidebar({ refreshKey, onSelect, selectedId, onClear }) {
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(false)

  const fetchList = async () => {
    setLoading(true)
    try {
      const data = await getHistory()
      setItems(data.history || [])
    } catch {
      setItems([])
    }
    setLoading(false)
  }

  useEffect(() => {
    fetchList()
  }, [refreshKey])

  const handleClick = async (id) => {
    try {
      const entry = await getHistoryDetail(id)
      onSelect(entry)
    } catch {
      // ignore
    }
  }

  const handleClear = async () => {
    await clearHistory()
    setItems([])
    onClear()
  }

  return (
    <>
      <div className="sidebar-header">
        <h3>📋 识别历史</h3>
        <p>点击查看详情</p>
      </div>
      <div className="sidebar-list">
        {items.length === 0 ? (
          <div className="history-empty">
            {loading ? "加载中..." : "暂无识别记录"}
          </div>
        ) : (
          items.map((item) => (
            <button
              key={item.id}
              className={`history-item ${selectedId === item.id ? "active" : ""}`}
              onClick={() => handleClick(item.id)}
            >
              <span className="hi-filename">{item.filename}</span>
              <span className="hi-top1">
                {item.top1 || "错误"} {item.top1_conf ? `${(item.top1_conf * 100).toFixed(1)}%` : ""}
              </span>
            </button>
          ))
        )}
      </div>
      <div className="sidebar-footer">
        <button
          className="btn btn-secondary btn-sm"
          disabled={items.length === 0}
          onClick={handleClear}
          style={{ width: "100%" }}
        >
          清空历史
        </button>
      </div>
    </>
  )
}
