const API = ""

export async function getStatus() {
  const r = await fetch(`${API}/api/status`)
  return r.json()
}

export async function getExamples() {
  const r = await fetch(`${API}/api/examples`)
  return r.json()
}

export async function predictSingle(file) {
  const fd = new FormData()
  fd.append("file", file)
  const r = await fetch(`${API}/api/predict/single`, {
    method: "POST",
    body: fd,
  })
  if (!r.ok) {
    const err = await r.text()
    throw new Error(err)
  }
  return r.json()
}

export async function predictBatch(files) {
  const fd = new FormData()
  for (const f of files) fd.append("files", f)
  const r = await fetch(`${API}/api/predict/batch`, {
    method: "POST",
    body: fd,
  })
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export async function getHistory() {
  const r = await fetch(`${API}/api/history`)
  return r.json()
}

export async function getHistoryDetail(id) {
  const r = await fetch(`${API}/api/history/${id}`)
  if (!r.ok) throw new Error("记录不存在")
  return r.json()
}

export async function clearHistory() {
  const r = await fetch(`${API}/api/history`, { method: "DELETE" })
  return r.json()
}
