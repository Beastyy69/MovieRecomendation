import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';
import './cinema.css';

const API = import.meta.env.VITE_API_URL || '/api';
const sessionId = localStorage.getItem('cinesense-session') || crypto.randomUUID();
localStorage.setItem('cinesense-session', sessionId);

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, { ...options, headers: { ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...options.headers } });
  if (!response.ok) throw new Error((await response.json().catch(() => ({}))).detail || 'Something went wrong');
  return response.json();
}

function Score({ value }) { return <span className="score">✦ {Number(value || 0).toFixed(1)}</span>; }

function MovieCard({ movie, onRecommend, onPreference, onTrailer, preferences }) {
  const [pending, setPending] = useState(false);
  const [media, setMedia] = useState(null);
  useEffect(() => { let active = true; getMedia(movie.title).then(data => { if (active) setMedia(data); }); return () => { active = false; }; }, [movie.title]);
  const choose = async (kind) => { setPending(true); try { await onPreference(movie.title, kind); } finally { setPending(false); } };
  const liked = preferences.some(p => p.title === movie.title && p.interaction === 'like');
  const seen = preferences.some(p => p.title === movie.title && p.interaction === 'watched');
  return <article className="movie-card">
    <div className="poster-fallback">{media?.poster_url ? <img src={media.poster_url} alt={`${movie.title} poster`} loading="lazy" onError={e => { e.currentTarget.style.display = 'none'; }} /> : <span>{movie.title.slice(0, 1)}</span>}</div>
    <div className="card-content">
      <div className="card-top"><Score value={movie.vote_average} /><span className="genre">{movie.genres?.slice(0, 2).join(' · ') || 'Film'}</span></div>
      <h3>{movie.title} {media?.year && <small>({media.year})</small>}</h3>
      <p>{movie.overview || 'No overview available for this title.'}</p>
      {movie.scores?.final !== undefined && <div className="match"><b>{Math.round(movie.scores.final * 100)}%</b> match <span>content {Math.round(movie.scores.content * 100)}% · your taste {Math.round(movie.scores.preference * 100)}%</span></div>}
      <div className="actions">
        <button className="primary" onClick={() => onRecommend(movie.title)}>Explore</button>
        <button disabled={pending || liked} className={liked ? 'selected' : ''} aria-label={`Like ${movie.title}`} aria-pressed={liked} onClick={() => choose('like')}>{liked ? '♥ Saved' : '♡ Like'}</button>
        <button disabled={pending || seen} className={seen ? 'selected' : ''} aria-label={`Mark ${movie.title} seen`} aria-pressed={seen} onClick={() => choose('watched')}>{seen ? '✓ Seen' : 'Seen'}</button>
        <button className="ghost" onClick={() => onTrailer(movie.title)}>Trailer</button>
      </div>
    </div>
  </article>;
}

const mediaRequests = new Map();
function getMedia(title) {
  if (!mediaRequests.has(title)) mediaRequests.set(title, request(`/media/${encodeURIComponent(title)}`).catch(() => ({})));
  return mediaRequests.get(title);
}

function App() {
  const [health, setHealth] = useState(null);
  const [query, setQuery] = useState('');
  const [search, setSearch] = useState([]);
  const [selected, setSelected] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [chat, setChat] = useState('');
  const [messages, setMessages] = useState([]);
  const [trailer, setTrailer] = useState(null);
  const [error, setError] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [featured, setFeatured] = useState([]);
  const [preferences, setPreferences] = useState([]);
  const [notice, setNotice] = useState('');
  const [heroMedia, setHeroMedia] = useState(null);
  const [selectedMovie, setSelectedMovie] = useState(null);
  const sectionRef = useRef(null);

  useEffect(() => { request('/health').then(setHealth).catch(e => setError(`Backend connect nahi hua: ${e.message}`)); }, []);
  useEffect(() => { request(`/preferences/${sessionId}`).then(setPreferences).catch(e => setError(`Preferences load nahi hui: ${e.message}`)); getMedia('Avatar').then(setHeroMedia); }, []);
  useEffect(() => { const close = e => { if (e.key === 'Escape') setTrailer(null); }; window.addEventListener('keydown', close); return () => window.removeEventListener('keydown', close); }, []);
  useEffect(() => { Promise.all(['Avatar', 'Toy Story', 'Interstellar', 'The Dark Knight'].map(title => request(`/movies/search?q=${encodeURIComponent(title)}&limit=1`).then(items => items[0]).catch(() => null))).then(items => setFeatured(items.filter(Boolean))); }, []);
  useEffect(() => {
    if (!query.trim()) return setSearch([]);
    const timer = setTimeout(() => request(`/movies/search?q=${encodeURIComponent(query)}`).then(setSearch).catch(e => setError(e.message)), 250);
    return () => clearTimeout(timer);
  }, [query]);
  const headline = useMemo(() => selected ? `Because you chose ${selected}` : 'Find your next favourite', [selected]);
  async function explore(title) {
    setSelected(title); setSelectedMovie(null); setRecommendations([]); setLoading(true); setError(''); setQuery('');
    sectionRef.current?.scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
    try { const [results, matches] = await Promise.all([request(`/recommendations/${encodeURIComponent(title)}?session_id=${sessionId}&limit=12`), request(`/movies/search?q=${encodeURIComponent(title)}&limit=20`)]); setRecommendations(results); setSelectedMovie(matches.find(m => m.title.toLowerCase() === title.toLowerCase()) || null); }
    catch (e) { setError(`Recommendations load nahi hui: ${e.message}`); } finally { setLoading(false); }
  }
  async function preference(title, interaction) {
    try { await request('/preferences', { method: 'POST', body: JSON.stringify({ session_id: sessionId, title, interaction }) }); setPreferences(old => [...old.filter(p => !(p.title === title && p.interaction === interaction)), {title, interaction}]); setNotice(interaction === 'watched' ? `${title} marked seen. It will be hidden from future recommendations.` : `${title} added to your likes.`); return true; }
    catch (e) { setError(e.message); return false; }
  }
  async function showTrailer(title) {
    setTrailer({ title, loading: true });
    try { setTrailer({ title, ...(await getMedia(title)) }); }
    catch { setTrailer({ title, trailer_url: null }); }
  }
  async function ask(event) {
    event.preventDefault(); if (!chat.trim()) return;
    const userMessage = chat; setChat(''); setMessages(m => [...m, { role: 'user', text: userMessage }]);
    setChatLoading(true);
    try { const data = await request('/assistant/chat', { method: 'POST', body: JSON.stringify({ session_id: sessionId, message: userMessage }) }); setMessages(m => [...m, { role: 'ai', text: data.reply }]); setRecommendations(data.recommendations); setSelected('your AI request'); }
    catch (e) { setError(e.message); } finally { setChatLoading(false); }
  }
  return <main>
    <nav><div className="brand">Cine<span>Sense</span></div><div className="status"><i className={health?.ollama_online ? 'online' : ''} />{health?.ollama_online ? 'Gemma 2B local' : 'Recommendation mode'}</div></nav>
    <section className="hero"><div className="hero-backdrop" aria-hidden="true" style={heroMedia?.backdrop_url ? {backgroundImage: `linear-gradient(90deg, #10151b 8%, #10151bb8 55%, #10151b55), url("${heroMedia.backdrop_url}")`} : undefined}/><div className="hero-content"><div className="eyebrow">YOUR LOCAL MOVIE COMPANION</div><h1>Discover movies<br/><em>that feel like you.</em></h1><p>Content intelligence, your personal taste, and local Gemma AI—without a cloud AI key.</p><div className="quick"><button disabled={loading} onClick={() => explore('Avatar')}>Explore Avatar</button><button disabled={loading} onClick={() => explore('Toy Story')}>Explore Toy Story</button><button disabled={loading} onClick={() => explore('Interstellar')}>Sci-Fi night</button></div></div></section>
    <section className="search-panel"><label htmlFor="movie-search">Search a movie</label><div className="search-box"><span>⌕</span><input id="movie-search" value={query} onChange={e => setQuery(e.target.value)} placeholder="Avatar, Toy Story, Interstellar…" /></div>{search.length > 0 && <div className="search-results">{search.map(movie => <button key={movie.title} onClick={() => explore(movie.title)}><span>{movie.title}</span><small>{movie.genres?.slice(0, 2).join(' · ')}</small></button>)}</div>}</section>
    {error && <div className="error">{error}<button onClick={() => setError('')}>×</button></div>}
    {notice && <div className="notice" role="status">{notice}<button aria-label="Dismiss" onClick={() => setNotice('')}>×</button></div>}
    <section className="discover" ref={sectionRef}><div className="section-heading"><div><div className="eyebrow">PERSONALISED DISCOVERY</div><h2>{headline}</h2></div>{loading && <span className="loader">Finding your match…</span>}</div>
      {selectedMovie && <div className="selected-details"><strong>{selectedMovie.title}</strong><span> ★ {selectedMovie.vote_average?.toFixed(1)} · {selectedMovie.genres?.join(' · ')}</span><p>{selectedMovie.overview}</p></div>}
      {loading ? <div className="film-loader" role="status"><div className="reel"><i/><i/><i/><i/></div><p>Finding your next story<span>...</span></p></div> : selected && !recommendations.length ? <p>No recommendations found. Try another movie.</p> : <div className="movie-grid">{(selected ? recommendations : featured).map((movie, i) => <MovieCard key={`${movie.title}-${i}`} movie={movie} preferences={preferences} onRecommend={explore} onPreference={preference} onTrailer={showTrailer} />)}</div>}
    </section>
    <section className="assistant"><div><div className="eyebrow">LOCAL GEMMA ASSISTANT</div><h2>Ask it your mood.</h2><p>Try Hinglish: “Mujhe fast-paced sci-fi chahiye, romance kam ho.”</p></div><div className="chat-panel">{messages.length > 0 && <div className="messages">{messages.map((m, i) => <div key={i} className={`message ${m.role}`}>{m.text}</div>)}</div>}{chatLoading && <p role="status">Gemma is thinking…</p>}<form onSubmit={ask}><input aria-label="Ask the movie assistant" value={chat} onChange={e => setChat(e.target.value)} placeholder="What are you in the mood for?" /><button type="submit" disabled={chatLoading}>Ask AI</button></form></div></section>
    {trailer && <div className="modal" role="dialog" aria-modal="true"><div className="modal-content"><button className="close" onClick={() => setTrailer(null)}>×</button><h2>{trailer.title}</h2>{trailer.loading ? <p>Loading trailer…</p> : trailer.trailer_url ? <iframe title={`${trailer.title} trailer`} src={trailer.trailer_url} allow="autoplay; encrypted-media" /> : <div className="no-trailer"><span>▻</span><p>No trailer is available. Add a TMDB API key in `.env` to enable media.</p></div>}</div></div>}
  </main>;
}
createRoot(document.getElementById('root')).render(<App />);
