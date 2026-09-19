import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import './App.css';

const API_BASE = 'https://deepseek-rag-chatbot.onrender.com';

function App() {
  const [messages, setMessages] = useState([
    { text: "Hi! Ask me anything — I can search your documents, browse the web, or do math.", sender: 'bot' }
  ]);
  const [input, setInput] = useState('');
  const [files, setFiles] = useState([]);
  const [status, setStatus] = useState('');
  const [loading, setLoading] = useState(false);
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'light');
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, loading]);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const newMessages = [...messages, { text: input, sender: 'user' }];
    setMessages(newMessages);
    setInput('');
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/api/chat`, { message: input });
      setMessages([...newMessages, { text: res.data.response, sender: 'bot' }]);
    } catch (e) {
      setMessages([...newMessages, { text: "Error connecting to server.", sender: 'bot' }]);
    }
    setLoading(false);
  };

  const uploadFiles = async () => {
    if (files.length === 0) return;
    const formData = new FormData();
    for (const f of files) formData.append('files', f);
    setStatus('Uploading...');
    await axios.post(`${API_BASE}/api/upload`, formData);
    setStatus('Uploaded. Re-indexing...');
    const res = await axios.post(`${API_BASE}/api/reindex`);
    setStatus(`Indexed ${res.data.chunks_indexed} chunks.`);
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <h2 className="sidebar-title">Documents</h2>
        <p className="sidebar-hint">Upload files, then re-index so the bot can search them.</p>

        <label className="file-input">
          <input type="file" multiple accept=".pdf,.txt,.md" onChange={(e) => setFiles([...e.target.files])} />
          {files.length ? `${files.length} file(s) selected` : 'Choose files'}
        </label>

        <button className="btn-primary" onClick={uploadFiles}>Upload &amp; Re-index</button>
        {status && <p className="status-text">{status}</p>}
      </aside>

      <main className="chat-panel">
        <header className="chat-header">
          <div>
            <span className="chat-title">AI Chat</span>
            <span className="chat-subtitle">Documents · Web search · Calculator</span>
          </div>
          <select
            className="theme-select"
            value={theme}
            onChange={(e) => setTheme(e.target.value)}
          >
            <option value="light">☀️ Light</option>
            <option value="dark">🌙 Dark</option>
            <option value="ocean">🌊 Ocean</option>
            <option value="sunset">🌅 Sunset</option>
          </select>
        </header>
        <div className="chat-scroll" ref={scrollRef}>
          {messages.map((m, i) => (
            <div key={i} className={`bubble-row ${m.sender}`}>
              <div className={`bubble ${m.sender}`}>
                <ReactMarkdown>{m.text}</ReactMarkdown>
              </div>
            </div>
          ))}
          {loading && (
            <div className="bubble-row bot">
              <div className="bubble bot typing">Thinking…</div>
            </div>
          )}
        </div>

        <div className="chat-input-row">
          <input
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Type a message..."
          />
          <button className="btn-primary send-btn" onClick={sendMessage}>Send</button>
        </div>
      </main>
    </div>
  );
}

export default App;
