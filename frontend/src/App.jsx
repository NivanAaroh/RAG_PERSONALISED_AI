import { useState } from 'react'
import ReactMarkdown from 'react-markdown'

const API_BASE = 'http://127.0.0.1:8000'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState('No document selected.')
  const [uploadError, setUploadError] = useState('')
  const [document, setDocument] = useState(null)

  const [ragQuery, setRagQuery] = useState('')
  const [ragLoading, setRagLoading] = useState(false)
  const [ragError, setRagError] = useState('')
  const [ragAnswer, setRagAnswer] = useState('')
  const [ragSources, setRagSources] = useState([])

  function handleFileChange(event) {
    const file = event.target.files?.[0] || null

    setSelectedFile(null)
    setDocument(null)
    setUploadError('')
    setUploadStatus('No document selected.')
    setRagAnswer('')
    setRagSources([])
    setRagError('')

    if (!file) return

    if (
      file.type !== 'application/pdf' &&
      !file.name.toLowerCase().endsWith('.pdf')
    ) {
      setUploadError('Only PDF files are supported.')
      return
    }

    setSelectedFile(file)
    setUploadStatus('PDF selected. Ready to upload.')
  }

  async function uploadDocument() {
    if (!selectedFile || uploading) return

    setUploading(true)
    setUploadError('')
    setDocument(null)
    setRagAnswer('')
    setRagSources([])
    setRagError('')
    setUploadStatus('Uploading...')

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      setUploadStatus('Indexing...')

      const res = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        let detail = `HTTP error: ${res.status}`

        try {
          const errorData = await res.json()
          if (errorData.detail) detail = errorData.detail
        } catch {
          // Keep HTTP error.
        }

        throw new Error(detail)
      }

      const data = await res.json()

      setDocument(data)
      setUploadStatus('Indexed successfully. Status: Ready.')
    } catch (error) {
      setUploadError(error.message)
      setUploadStatus('Upload failed.')
    } finally {
      setUploading(false)
    }
  }

  async function askRagQuestion() {
    const question = ragQuery.trim()

    if (!question || ragLoading || !document?.document_id) return

    setRagLoading(true)
    setRagError('')
    setRagAnswer('')
    setRagSources([])

    try {
      const res = await fetch(`${API_BASE}/api/rag`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question,
          document_id: document.document_id,
          top_k: 5,
        }),
      })

      if (!res.ok) {
        let detail = `HTTP error: ${res.status}`

        try {
          const errorData = await res.json()
          if (errorData.detail) detail = errorData.detail
        } catch {
          // Keep HTTP error.
        }

        throw new Error(detail)
      }

      if (!res.body) {
        throw new Error('No response stream received from backend')
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { value, done } = await reader.read()

        if (done) break

        buffer += decoder.decode(value, { stream: true })

        const events = buffer.split('\n\n')
        buffer = events.pop() || ''

        for (const eventBlock of events) {
          processSseEvent(eventBlock)
        }
      }

      buffer += decoder.decode()

      if (buffer.trim()) {
        processSseEvent(buffer)
      }
    } catch (error) {
      setRagError(`RAG request failed: ${error.message}`)
    } finally {
      setRagLoading(false)
    }
  }

  function processSseEvent(eventBlock) {
    const lines = eventBlock.split('\n')
    let eventName = ''
    let dataText = ''

    for (const line of lines) {
      if (line.startsWith('event:')) {
        eventName = line.slice(6).trim()
      } else if (line.startsWith('data:')) {
        dataText += line.slice(5).trim()
      }
    }

    if (!eventName || !dataText) return

    try {
      const data = JSON.parse(dataText)

      if (eventName === 'sources') {
        setRagSources(data)
      } else if (eventName === 'answer') {
        setRagAnswer((previous) => previous + data)
      } else if (eventName === 'done') {
        // Stream completed.
      }
    } catch {
      setRagError('Invalid SSE data received from backend.')
    }
  }

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-6">
      <div className="w-full max-w-4xl rounded-2xl bg-white p-8 shadow-lg">
        <h1 className="text-3xl font-bold text-blue-600">
          RAG Personalised AI
        </h1>

        <p className="mt-2 text-gray-600">
          Part 6 — Browser Vertical Slice (WP6-B RAG Flow)
        </p>

        <div className="mt-6 rounded-lg border border-gray-200 p-4">
          <h2 className="text-lg font-semibold text-gray-800">
            Upload PDF
          </h2>

          <input
            type="file"
            accept=".pdf,application/pdf"
            onChange={handleFileChange}
            disabled={uploading}
            className="mt-3 w-full text-sm"
          />

          <button
            onClick={uploadDocument}
            disabled={uploading || !selectedFile}
            className="mt-4 rounded-lg bg-green-600 px-5 py-3 font-medium text-white hover:bg-green-700 disabled:opacity-50"
          >
            {uploading ? 'Indexing...' : 'Upload PDF'}
          </button>

          <p className="mt-3 text-sm text-gray-700">
            <span className="font-semibold">Status:</span>{' '}
            {uploadStatus}
          </p>

          {uploadError && (
            <p className="mt-2 text-sm text-red-600">{uploadError}</p>
          )}

          {document && (
            <div className="mt-4 rounded-lg bg-gray-100 p-4 text-left text-sm text-gray-800">
              <p><span className="font-semibold">Status:</span> Ready</p>
              <p className="mt-1"><span className="font-semibold">Document:</span> {document.document_name}</p>
              <p className="mt-1"><span className="font-semibold">Document ID:</span> {document.document_id}</p>
              <p className="mt-1"><span className="font-semibold">Pages:</span> {document.page_count}</p>
              <p className="mt-1"><span className="font-semibold">Pages with text:</span> {document.pages_with_text}</p>
              <p className="mt-1"><span className="font-semibold">Chunks:</span> {document.chunk_count}</p>
              <p className="mt-1"><span className="font-semibold">Embeddings:</span> {document.embedding_count}</p>
              <p className="mt-1"><span className="font-semibold">Stored:</span> {document.stored_count}</p>
            </div>
          )}
        </div>

        <div className="mt-6 rounded-lg border border-blue-200 p-4">
          <h2 className="text-lg font-semibold text-gray-800">
            Ask Document (RAG)
          </h2>

          <input
            type="text"
            value={ragQuery}
            onChange={(event) => setRagQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') askRagQuestion()
            }}
            placeholder="Enter a question about the uploaded document"
            disabled={ragLoading || !document}
            className="mt-3 w-full rounded-lg border border-gray-300 px-4 py-3 outline-none focus:border-blue-500 disabled:opacity-50"
          />

          <button
            onClick={askRagQuestion}
            disabled={ragLoading || !document || !ragQuery.trim()}
            className="mt-4 rounded-lg bg-blue-600 px-5 py-3 font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            {ragLoading ? 'Generating...' : 'Ask RAG'}
          </button>

          {!document && (
            <p className="mt-3 text-sm text-gray-500">
              Upload a PDF before asking questions.
            </p>
          )}

          {ragError && (
            <p className="mt-3 text-sm text-red-600">{ragError}</p>
          )}

          {ragAnswer && (
            <div className="mt-6 rounded-lg bg-gray-50 p-4 text-left">
              <h3 className="text-md font-semibold text-gray-800">
                Answer
              </h3>

              <div className="mt-2 text-gray-900">
                <ReactMarkdown>{ragAnswer}</ReactMarkdown>
              </div>
            </div>
          )}

          {ragSources.length > 0 && (
            <div className="mt-6 text-left">
              <h3 className="text-md font-semibold text-gray-800">
                Sources ({ragSources.length})
              </h3>

              <div className="mt-4 space-y-4">
                {ragSources.map((source, index) => (
                  <div
                    key={source.chunk_id || index}
                    className="rounded-lg border border-gray-200 p-4"
                  >
                    <p className="text-sm text-gray-800">
                      <span className="font-semibold">Source:</span>{' '}
                      {index + 1}
                    </p>

                    <p className="mt-1 text-sm text-gray-800">
                      <span className="font-semibold">Page:</span>{' '}
                      {source.page}
                    </p>

                    <p className="mt-1 text-sm text-gray-800">
                      <span className="font-semibold">Chunk ID:</span>{' '}
                      {source.chunk_id}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
