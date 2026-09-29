import { useState } from 'react'
import ReactMarkdown from 'react-markdown'

const API_BASE = 'http://127.0.0.1:8000'

function App() {
  const [message, setMessage] = useState('')
  const [response, setResponse] = useState('')
  const [loading, setLoading] = useState(false)

  const [selectedFile, setSelectedFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState('No document selected.')
  const [uploadError, setUploadError] = useState('')
  const [document, setDocument] = useState(null)

  const [retrieveQuery, setRetrieveQuery] = useState('')
  const [retrieveLoading, setRetrieveLoading] = useState(false)
  const [retrieveError, setRetrieveError] = useState('')
  const [retrieval, setRetrieval] = useState(null)

  function handleFileChange(event) {
    const file = event.target.files?.[0] || null

    setSelectedFile(null)
    setDocument(null)
    setUploadError('')
    setUploadStatus('No document selected.')
    setRetrieval(null)
    setRetrieveError('')

    if (!file) {
      return
    }

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
    if (!selectedFile || uploading) {
      return
    }

    setUploading(true)
    setUploadError('')
    setDocument(null)
    setRetrieval(null)
    setRetrieveError('')
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

          if (errorData.detail) {
            detail = errorData.detail
          }
        } catch {
          // Keep the HTTP error message.
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

  async function retrieveEvidence() {
    const trimmedQuery = retrieveQuery.trim()

    if (!trimmedQuery || retrieveLoading || !document?.document_id) {
      return
    }

    setRetrieveLoading(true)
    setRetrieveError('')
    setRetrieval(null)

    try {
      const res = await fetch(`${API_BASE}/api/retrieve`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: trimmedQuery,
          document_id: document.document_id,
          top_k: 5,
        }),
      })

      if (!res.ok) {
        let detail = `HTTP error: ${res.status}`

        try {
          const errorData = await res.json()

          if (errorData.detail) {
            detail = errorData.detail
          }
        } catch {
          // Keep the HTTP error message.
        }

        throw new Error(detail)
      }

      const data = await res.json()
      setRetrieval(data)
    } catch (error) {
      setRetrieveError(`Retrieval failed: ${error.message}`)
    } finally {
      setRetrieveLoading(false)
    }
  }

  async function sendMessage() {
    const trimmedMessage = message.trim()

    if (!trimmedMessage || loading) {
      return
    }

    setLoading(true)
    setResponse('')

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: trimmedMessage,
        }),
      })

      if (!res.ok) {
        throw new Error(`HTTP error: ${res.status}`)
      }

      if (!res.body) {
        throw new Error('No response stream received from backend')
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let receivedText = ''

      while (true) {
        const { value, done } = await reader.read()

        if (done) {
          break
        }

        const chunk = decoder.decode(value, { stream: true })

        if (chunk) {
          receivedText += chunk
          setResponse(receivedText)
        }
      }

      const finalChunk = decoder.decode()

      if (finalChunk) {
        receivedText += finalChunk
        setResponse(receivedText)
      }

      if (!receivedText.trim()) {
        throw new Error('Empty response received from backend')
      }
    } catch (error) {
      setResponse(`Error: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-6">
      <div className="w-full max-w-4xl rounded-2xl bg-white p-8 shadow-lg">
        <h1 className="text-3xl font-bold text-blue-600">
          RAG Personalised AI
        </h1>

        <p className="mt-2 text-gray-600">
          Part 6 — Browser Vertical Slice
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
            <p className="mt-2 text-sm text-red-600">
              {uploadError}
            </p>
          )}

          {document && (
            <div className="mt-4 rounded-lg bg-gray-100 p-4 text-left text-sm text-gray-800">
              <p>
                <span className="font-semibold">Status:</span> Ready
              </p>

              <p className="mt-1">
                <span className="font-semibold">Document:</span>{' '}
                {document.document_name}
              </p>

              <p className="mt-1">
                <span className="font-semibold">Document ID:</span>{' '}
                {document.document_id}
              </p>

              <p className="mt-1">
                <span className="font-semibold">Pages:</span>{' '}
                {document.page_count}
              </p>

              <p className="mt-1">
                <span className="font-semibold">Pages with text:</span>{' '}
                {document.pages_with_text}
              </p>

              <p className="mt-1">
                <span className="font-semibold">Chunks:</span>{' '}
                {document.chunk_count}
              </p>

              <p className="mt-1">
                <span className="font-semibold">Embeddings:</span>{' '}
                {document.embedding_count}
              </p>

              <p className="mt-1">
                <span className="font-semibold">Stored:</span>{' '}
                {document.stored_count}
              </p>
            </div>
          )}
        </div>

        <div className="mt-6 rounded-lg border border-blue-200 p-4">
          <h2 className="text-lg font-semibold text-gray-800">
            Retrieve Evidence
          </h2>

          <p className="mt-1 text-sm text-gray-600">
            Existing diagnostic retrieval interface. RAG question flow will be
            added in WP6-B.
          </p>

          <input
            type="text"
            value={retrieveQuery}
            onChange={(event) => setRetrieveQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                retrieveEvidence()
              }
            }}
            placeholder="Enter a question about the uploaded document"
            disabled={retrieveLoading || !document}
            className="mt-3 w-full rounded-lg border border-gray-300 px-4 py-3 outline-none focus:border-blue-500 disabled:opacity-50"
          />

          <button
            onClick={retrieveEvidence}
            disabled={
              retrieveLoading ||
              !document ||
              !retrieveQuery.trim()
            }
            className="mt-4 rounded-lg bg-blue-600 px-5 py-3 font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            {retrieveLoading ? 'Retrieving...' : 'Retrieve'}
          </button>

          {!document && (
            <p className="mt-3 text-sm text-gray-500">
              Upload a PDF before running retrieval.
            </p>
          )}

          {retrieveError && (
            <p className="mt-3 text-sm text-red-600">
              {retrieveError}
            </p>
          )}

          {retrieval && (
            <div className="mt-6 text-left">
              <div className="rounded-lg bg-gray-100 p-4 text-sm text-gray-800">
                <p>
                  <span className="font-semibold">Query:</span>{' '}
                  {retrieval.query}
                </p>

                <p className="mt-1">
                  <span className="font-semibold">Document ID:</span>{' '}
                  {retrieval.document_id}
                </p>

                <p className="mt-1">
                  <span className="font-semibold">Top-K:</span>{' '}
                  {retrieval.top_k}
                </p>
              </div>

              <div className="mt-4 space-y-4">
                {retrieval.results?.map((result, index) => (
                  <div
                    key={result.chunk_id}
                    className="rounded-lg border border-gray-200 p-4"
                  >
                    <div className="text-sm text-gray-800">
                      <p>
                        <span className="font-semibold">Result:</span>{' '}
                        {index + 1}
                      </p>

                      <p className="mt-1">
                        <span className="font-semibold">Chunk ID:</span>{' '}
                        {result.chunk_id}
                      </p>

                      <p className="mt-1">
                        <span className="font-semibold">Page:</span>{' '}
                        {result.page}
                      </p>

                      <p className="mt-1">
                        <span className="font-semibold">Distance:</span>{' '}
                        {result.distance}
                      </p>
                    </div>

                    <pre className="mt-3 whitespace-pre-wrap rounded-lg bg-gray-100 p-3 text-sm text-gray-800">
                      {result.text}
                    </pre>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="mt-6">
          <input
            type="text"
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                sendMessage()
              }
            }}
            placeholder="Enter a message"
            disabled={loading}
            className="w-full rounded-lg border border-gray-300 px-4 py-3 outline-none focus:border-blue-500 disabled:opacity-50"
          />

          <button
            onClick={sendMessage}
            disabled={loading || !message.trim()}
            className="mt-4 rounded-lg bg-blue-600 px-5 py-3 font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? 'Generating...' : 'Send'}
          </button>
        </div>

        {response && (
          <div className="mt-6 rounded-lg bg-gray-100 p-4">
            <p className="text-sm font-semibold text-gray-700">
              Backend response:
            </p>

            <div className="mt-2 text-left text-gray-900">
              <ReactMarkdown>{response}</ReactMarkdown>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App