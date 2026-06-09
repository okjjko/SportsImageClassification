import { useState } from "react"
import "./App.css"
import Header from "./components/Header"
import Sidebar from "./components/Sidebar"
import SingleTab from "./components/SingleTab"
import BatchTab from "./components/BatchTab"
import DetailPanel from "./components/DetailPanel"
import Footer from "./components/Footer"

export default function App() {
  const [activeTab, setActiveTab] = useState("single")
  const [historyList, setHistoryList] = useState([])
  const [selectedEntry, setSelectedEntry] = useState(null)
  const [refreshKey, setRefreshKey] = useState(0)

  const refreshHistory = () => setRefreshKey((k) => k + 1)

  const handleSelect = (entry) => {
    setSelectedEntry(entry)
  }

  const handleBack = () => {
    setSelectedEntry(null)
    refreshHistory()
  }

  const isDetail = selectedEntry !== null

  return (
    <>
      <div className="sidebar">
        <Sidebar
          refreshKey={refreshKey}
          onSelect={handleSelect}
          selectedId={selectedEntry?.id ?? null}
          onClear={refreshHistory}
        />
      </div>
      <div className="app-container">
        <Header />
        <div className="main-layout">
          <div className="content-area">
            {isDetail ? (
              <DetailPanel entry={selectedEntry} onBack={handleBack} />
            ) : (
              <>
                <div className="tabs">
                  <button
                    className={`tab-btn ${activeTab === "single" ? "active" : ""}`}
                    onClick={() => setActiveTab("single")}
                  >
                    单张识别
                  </button>
                  <button
                    className={`tab-btn ${activeTab === "batch" ? "active" : ""}`}
                    onClick={() => setActiveTab("batch")}
                  >
                    批量分类
                  </button>
                </div>

                {activeTab === "single" && (
                  <SingleTab onPredictionDone={refreshHistory} />
                )}
                {activeTab === "batch" && (
                  <BatchTab onPredictionDone={refreshHistory} />
                )}
              </>
            )}
          </div>
        </div>
        <Footer />
      </div>
    </>
  )
}
