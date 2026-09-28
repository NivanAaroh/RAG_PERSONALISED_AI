import { useState } from 'react'

function App() {
  const [message, setMessage] = useState('')
  const [response, setResponse] = useState('')
  const [loading, setLoading] = useState(false)

  async function sendMessage() {
    if (!message.trim()) {
      return
    }

    setLoading(true)
    setResponse('')

    try {
      const res = await fetch(
        `http://127.0.0.1:8000/chat?message=${encodeURIComponent(message)}`,
        {
          method: 'POST',
        }

      )

      if (!res.ok) {
        throw new Error(`HTTP error: ${res.status}`)
      }

      const data = await res.json()
      setResponse(data.response)
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
            className="w-full rounded-lg border border-gray-300 px-4 py-3 outline-none focus:border-blue-500"
          />

          <button
            onClick={sendMessage}
            disabled={loading}
            className="mt-4 rounded-lg bg-blue-600 px-5 py-3 font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? 'Sending...' : 'Send'}
          </button>
        </div>

        {response && (
          <div className="mt-6 rounded-lg bg-gray-100 p-4">
            <p className="text-sm font-semibold text-gray-700">
              Backend response:
            </p>

            <p className="mt-2 text-gray-900">
              {response}
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

export default App