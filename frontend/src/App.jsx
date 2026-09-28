import { useState } from 'react'
import ReactMarkdown from 'react-markdown'

function App() {
  const [message, setMessage] = useState('')
  const [response, setResponse] = useState('')
  const [loading, setLoading] = useState(false)

  async function sendMessage() {
    const trimmedMessage = message.trim()

    if (!trimmedMessage || loading) {
      return
    }

    setLoading(true)
    setResponse('')

    try {
      const res = await fetch('http://127.0.0.1:8000/chat', {
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
      <div className="w-full max-w-xl rounded-2xl bg-white p-8 shadow-lg">
        <h1 className="text-3xl font-bold text-blue-600">
          RAG Personalised AI
        </h1>

        <p className="mt-2 text-gray-600">
          Part 0 — Frontend to Backend Test
        </p>

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