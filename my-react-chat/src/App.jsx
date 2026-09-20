import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import './App.css';
import { onAuthStateChanged, signOut } from 'firebase/auth';
import { auth } from './firebase';
import Login from './Login';

const API_BASE = import.meta.env.VITE_API_BASE || 'https://deepseek-rag-chatbot.onrender.com';

function App() {
  const [messages, setMessages] = useState([
    { text: "Hi! Ask me anything — I can search your documents, browse the web, or do math.", sender: 'bot' }
  ]);
  const [input, setInput] = useState('');
  const [files, setFiles] = useState([]);
  const [status, setStatus] = useState('');
  const [loading, setLoading] = useState(false);
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'light');
  const [databases, setDatabases] = useState([]);
  const [selectedDb, setSelectedDb] = useState('f1');
  const [dbFile, setDbFile] = useState(null);
  const [dbStatus, setDbStatus] = useState('');
  const scrollRef = useRef(null);
  const [user, setUser] = useState(null);
  const [authReady, setAuthReady] = useState(false);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, loading]);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
  }, [theme]);

  useEffect(() => {
    return onAuthStateChanged(auth, (u) => {
      setUser(u);
      setAuthReady(true);
    });
  }, []);

  useEffect(() => {
    if (!user) return;
    axios.get(`${API_BASE}/api/databases`).then((res) => {
      setDatabases(res.data.databases);
    }).catch(() => {});
  }, [user]);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const newMessages = [...messages, { text: input, sender: 'user' }];
    setMessages(newMessages);
    setInput('');
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/api/chat`, { message: input, db_id: selectedDb });
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

  const uploadDatabase = async () => {
    if (!dbFile) return;
    const formData = new FormData();
    formData.append('file', dbFile);
    setDbStatus('Uploading...');
    const res = await axios.post(`${API_BASE}/api/databases/upload`, formData);
    if (res.data.error) {
      setDbStatus(res.data.error);
      return;
    }
    setDatabases(res.data.databases);
    setSelectedDb(res.data.saved.replace(/\.(sqlite|db)$/i, ''));
    setDbStatus(`Added ${res.data.saved}.`);
  };

  if (!authReady) return <p style={{ padding: 20 }}>Loading auth…</p>;
  if (!user) return <Login />;

  return (
    <div className="app-shell">
      <div className="sidebar-stack">
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

        <aside className="sidebar">
          <h2 className="sidebar-title">Databases</h2>
          <p className="sidebar-hint">Select a database to query, or upload a new one.</p>

          <select
            className="theme-select"
            value={selectedDb}
            onChange={(e) => setSelectedDb(e.target.value)}
          >
            {databases.map((db) => (
              <option key={db.id} value={db.id}>{db.name}</option>
            ))}
          </select>

          <label className="file-input">
            <input type="file" accept=".sqlite,.db" onChange={(e) => setDbFile(e.target.files[0])} />
            {dbFile ? dbFile.name : 'Choose a database file'}
          </label>

          <button className="btn-primary" onClick={uploadDatabase}>Upload Database</button>
          {dbStatus && <p className="status-text">{dbStatus}</p>}
        </aside>
      </div>

      <main className="chat-panel">
        <header className="chat-header">
          <div>
            <span className="chat-title">AI Chat</span>
            <span className="chat-subtitle">Documents · Web search · Calculator</span>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <select
              className="theme-select"
              value={theme}
              onChange={(e) => setTheme(e.target.value)}
            >
              <option value="light">Light</option>
              <option value="dark">Dark</option>
              <option value="ocean">Ocean</option>
              <option value="sunset">Sunset</option>
            </select>
            <button className="btn-primary" onClick={() => signOut(auth)}>Sign out</button>
          </div>
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