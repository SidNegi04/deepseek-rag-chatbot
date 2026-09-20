import { useState } from 'react';
import { signInWithPopup } from 'firebase/auth';
import { auth, googleProvider } from './firebase';

export default function Login() {
  const [error, setError] = useState('');

  const handleGoogle = async () => {
    try {
      await signInWithPopup(auth, googleProvider);
    } catch (e) {
      if (e.code !== 'auth/popup-closed-by-user') {
        setError('Sign-in failed. Please try again.');
      }
    }
  };

  return (
    <div className="login-shell">
      <div className="sidebar login-card">
        <h2 className="sidebar-title">DeepSeek RAG Chatbot</h2>
        <p className="sidebar-hint">Sign in to continue.</p>
        <button className="btn-primary" onClick={handleGoogle}>Continue with Google</button>
        {error && <p className="status-text">{error}</p>}
      </div>
    </div>
  );
}