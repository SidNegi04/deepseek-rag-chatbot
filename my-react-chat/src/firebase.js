import axios from 'axios';

// ...existing code stays as is...

axios.interceptors.request.use(async (config) => {
  const user = auth.currentUser;
  if (user) {
    config.headers.Authorization = `Bearer ${await user.getIdToken()}`;
  }
  return config;
});
import { initializeApp } from 'firebase/app';
import { getAuth, GoogleAuthProvider } from 'firebase/auth';

const firebaseConfig = {
  apiKey: "AIzaSyANpEpBfBkwo1OtcN4vorwWUQ68KlSHSGI",
  authDomain: "deepseek-chatbot-b31c3.firebaseapp.com",
  projectId: "deepseek-chatbot-b31c3",
  appId: "1:1007983092397:web:1d17fc8e36a1f916fb9c24",
};

export const auth = getAuth(initializeApp(firebaseConfig));
export const googleProvider = new GoogleAuthProvider();