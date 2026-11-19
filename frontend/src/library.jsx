import React, {useEffect, useLayoutEffect, useRef, useState, useCallback, useMemo} from 'react';
import {createRoot} from 'react-dom/client';
import {motion, AnimatePresence, useReducedMotion} from 'framer-motion';
import './library.css';
import Magnet from './reactbits/Magnet';
import SpotlightCard from './reactbits/SpotlightCard';

const sid = localStorage.getItem('cinesense-session') || crypto.randomUUID();
localStorage.setItem('cinesense-session', sid);
const api = import.meta.env.VITE_API_URL || '/api';

async function request(path, options) {
  const timeout = AbortSignal.timeout(60000);
  const signal = options?.signal ? AbortSignal.any([options.signal, timeout]) : timeout;
  const response = await fetch(api + path, {...options, signal});
  if (!response.ok) {
    const error = Error(response.status === 404 ? 'Film not found in this catalog.' : 'The film service is unavailable. Please try again.');
    error.status = response.status;
    throw error;
  }
  return response.json();
}
function post(path, body) {
  return request(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
}

// --- Poster cache (deduplication) ---
const _posterPromises = new Map();
function fetchPoster(movieId) {
  if (!movieId) return Promise.resolve(null);
  if (!_posterPromises.has(movieId)) {
    _posterPromises.set(movieId, request(`/catalog/${encodeURIComponent(movieId)}/poster`)
      .then(d => d?.poster_url || null)
      .catch(() => null));
  }
  return _posterPromises.get(movieId);
}

// --- useResource hook ---
function useResource(path, delay=0) {
  const [state,set] = useState({});
  useEffect(() => {
    if (!path) return;
    const controller = new AbortController();
    const timer = setTimeout(() => request(path,{signal:controller.signal}).then(data => {
      if (!controller.signal.aborted) set({path,data});
    }).catch(err => {if (!controller.signal.aborted) set({path,error:err.message,status:err.status});}), delay);
    return () => {clearTimeout(timer); controller.abort();};
  },[path,delay]);
  return state.path === path ? state : {};
}

const href = movie => '/movie/' + encodeURIComponent(movie.movie_id);
const prefersReducedMotion = () => window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches;

// --- Motion variants ---
const fadeIn = {initial:{opacity:0}, animate:{opacity:1}, exit:{opacity:0}, transition:{duration:0.3}};
const slideUp = {initial:{opacity:0, y:20}, animate:{opacity:1, y:0}, exit:{opacity:0, y:-10}, transition:{duration:0.35}};
const staggerContainer = {animate:{transition:{staggerChildren:0.07}}};
const staggerChild = {initial:{opacity:0, y:16, scale:0.97}, animate:{opacity:1, y:0, scale:1}, transition:{duration:0.3}};
const hoverPoster = {scale:1.04, transition:{duration:0.3}};
const focusPoster = {scale:1.02, boxShadow:'0 0 0 3px var(--brass)', transition:{duration:0.2}};

// ==================== SmartPoster ====================
function SmartPoster({movie, className=""}) {
  const [src, setSrc] = useState(movie.poster_url);
  const [phase, setPhase] = useState('catalog'); // catalog -> tmdb -> placeholder
  const movieIdRef = useRef(movie.movie_id);
  const reducedMotion = useReducedMotion();
  useEffect(() => {
    let active = true;
    setSrc(movie.poster_url || null);
    setPhase('catalog');
    if (!movie.poster_url) {
      fetchPoster(movie.movie_id).then(url => {
        if (active) { setSrc(url); setPhase(url ? 'tmdb' : 'placeholder'); }
      });
    }
    return () => { active = false; };
  }, [movie.movie_id, movie.poster_url]);

  // Reset when movie changes
  useEffect(() => {
    if (movieIdRef.current !== movie.movie_id) {
      movieIdRef.current = movie.movie_id;
      setSrc(movie.poster_url);
      setPhase('catalog');
    }
  }, [movie.movie_id, movie.poster_url]);

  // Also reset if poster_url changes for same movie
  useEffect(() => {
    if (phase === 'catalog' && movie.poster_url && movie.poster_url !== src) {
      setSrc(movie.poster_url);
    }
  }, [movie.poster_url]);

  const handleError = useCallback(() => {
    if (phase === 'catalog') {
      setPhase('tmdb');
      fetchPoster(movie.movie_id).then(url => {
        if (movieIdRef.current !== movie.movie_id) return;
        if (url && url !== src) setSrc(url);
        else { setSrc(null); setPhase('placeholder'); }
      });
    } else {
      setSrc(null);
      setPhase('placeholder');
    }
  }, [phase, movie.movie_id, src]);

  const inner = src ? (
    <img
      loading="lazy"
      src={src}
      alt={movie.title + ' poster'}
      onError={handleError}
      style={{width:'100%', height:'100%', objectFit:'cover'}}
    />
  ) : (
    <div className="poster-placeholder">
      <span>{movie.title?.[0] || '?'}</span>
      <small>{movie.title}</small>
    </div>
  );

  return (
    <motion.div
      className={`poster ${className}`}
      whileHover={reducedMotion ? undefined : hoverPoster}
      whileFocus={reducedMotion ? undefined : focusPoster}
    >
      {inner}
    </motion.div>
  );
}

// ==================== Shelf ====================
function Shelf({movies, stagger=false}) {
  const reducedMotion = useReducedMotion();
  const Container = stagger && !reducedMotion ? motion.div : 'div';
  const containerProps = stagger && !reducedMotion ? {variants:staggerContainer, initial:"initial", animate:"animate"} : {};

  return (
    <Container className="shelf" {...containerProps}>
      {movies.map((movie, i) => {
        const inner = (
          <a key={movie.movie_id} href={href(movie)}>
            <SmartPoster movie={movie} />
            <small>{movie.year}　{movie.genres?.[0]}</small>
            <h3>{movie.title}</h3>
          </a>
        );
        if (stagger && !reducedMotion) {
          return <motion.div key={movie.movie_id} variants={staggerChild}>{inner}</motion.div>;
        }
        return <React.Fragment key={movie.movie_id}>{inner}</React.Fragment>;
      })}
    </Container>
  );
}

// ==================== Cinematic Background ====================
function CinematicBg({backdropUrl, posterUrl, movieId}) {
  const [loaded, setLoaded] = useState(false);
  const [currentSrc, setCurrentSrc] = useState(null);
  const reducedMotion = useReducedMotion();

  useEffect(() => {
    setLoaded(false);
    setCurrentSrc(null);
    let active = true;
    if (!backdropUrl && !posterUrl) return;
    const img = new Image();
    const target = backdropUrl || posterUrl;
    img.onload = () => { if (active) { setCurrentSrc(target); setLoaded(true); } };
    img.onerror = () => {
      if (target === backdropUrl && posterUrl) {
        const fallback = new Image();
        fallback.onload = () => { if (active) { setCurrentSrc(posterUrl); setLoaded(true); } };
        fallback.src = posterUrl;
      }
    };
    img.src = target;
    return () => { active = false; };
  }, [backdropUrl, posterUrl, movieId]);

  return (
    <>
      <AnimatePresence mode="wait">
        {loaded && currentSrc && (
          <motion.img
            key={currentSrc}
            className={reducedMotion ? "pan-static" : "pan"}
            alt=""
            src={currentSrc}
            initial={{opacity:0}}
            animate={{opacity:1}}
            exit={{opacity:0}}
            transition={{duration: reducedMotion ? 0 : 0.6}}
          />
        )}
      </AnimatePresence>
      <div className="veil" />
      <div className="grain" />
    </>
  );
}

// ==================== Hero ====================
function Hero({movie, media, home}) {
  return (
    <header>
      <CinematicBg
        backdropUrl={media?.backdrop_url}
        posterUrl={movie.poster_url}
        movieId={movie.movie_id}
      />
      <div className="inner">
        <small>{home ? 'FEATURED FILM' : movie.genres?.join(' · ')}　/　{movie.year}</small>
        <h1>{movie.title}</h1>
        {home && <Magnet padding={30} maxOffset={6}><a className="cta" href={href(movie)}>Explore film</a></Magnet>}
      </div>
    </header>
  );
}

// ==================== Search ====================
function Search() {
  const [query,setQuery] = useState(history.state?.query || '');
  useEffect(() => {
    const restore = () => { if(location.pathname==='/') setQuery(history.state?.query||''); };
    window.addEventListener('popstate', restore);
    return () => window.removeEventListener('popstate', restore);
  },[]);
  const result = useResource(query.trim() ? '/movies/search?q='+encodeURIComponent(query.trim()) : null, 180);
  return (
    <section>
      <label htmlFor="search">Search the catalog</label>
      <input id="search" type="search" placeholder="Title of a film…" value={query} onChange={e => {setQuery(e.target.value); history.replaceState({...history.state,query:e.target.value},'');}} />
      {query && (
        <div className="results">
          {result.error || (result.data ? (result.data.length ? result.data.map(m =>
            <a key={m.movie_id} href={href(m)}>
              <SmartPoster movie={m} className="result-poster" />
              <span>{m.title} <small>{m.year}</small></span>
            </a>
          ) : 'No matching films.') : 'Searching…')}
        </div>
      )}
    </section>
  );
}

// ==================== Markdown-safe text renderer ====================
function SafeText({text}) {
  if (!text) return null;
  // Strip markdown bold/italic markers, render clean text
  const clean = text
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/\*(.+?)\*/g, '$1')
    .replace(/__(.+?)__/g, '$1')
    .replace(/_(.+?)_/g, '$1')
    .replace(/^#+\s+/gm, '')
    .replace(/^[-*]\s+/gm, '• ');
  return <>{clean.split('\n').map((line, i) => <React.Fragment key={i}>{i > 0 && <br/>}{line}</React.Fragment>)}</>;
}

// ==================== AI Assistant ====================
function Assistant() {
  const [text, setText] = useState('');
  const [reply, setReply] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const resultsRef = useRef(null);
  const reducedMotion = useReducedMotion();

  async function submit(e) {
    e.preventDefault();
    if (!text.trim() || busy) return;
    setBusy(true);
    setError('');
    try {
      const data = await post('/assistant/chat', {session_id: sid, message: text});
      setReply(data);
      // Scroll results into view on mobile
      setTimeout(() => resultsRef.current?.scrollIntoView({behavior: prefersReducedMotion() ? 'auto' : 'smooth', block: 'start'}), 100);
    } catch (err) {
      setError(err.message);
      setReply(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="ai-section"><SpotlightCard spotlightColor="#d0b98b" intensity={0.10} borderGlow={0.15} className="assistant-spotlight">
      <h2>Ask CineSense</h2>
      <p className="ai-hint">Try: "Horror type kuch", "Feel-good family movie", "Mujhe dark sci-fi chahiye"</p>
      <form onSubmit={submit} className="ai-form">
        <input aria-label="Movie request" placeholder="Aaj kya dekhna hai?" value={text} onChange={e => setText(e.target.value)} />
        <button disabled={busy}>{busy ? 'Thinking…' : 'Ask Gemma'}</button>
      </form>

      {/* Loading state */}
      {busy && (
        <div className="ai-loading" role="status">
          <div className="ai-spinner" />
          <p>Finding movies that match your mood…</p>
        </div>
      )}

      {/* Error state */}
      {error && <p className="ai-error" role="alert">{error}</p>}

      {/* Results */}
      {reply && !busy && (
        <div ref={resultsRef} className="ai-results">
          {/* Explanation */}
          <div className="ai-explanation" role="status">
            {reply.detected_genres?.length > 0 && (
              <div className="ai-genres">
                {reply.detected_genres.map(g => <span key={g} className="genre-tag">{g}</span>)}
              </div>
            )}
            <SafeText text={reply.reply} />
            {!reply.ai_available && <small className="ai-offline">⚠ Gemma AI offline — showing catalog results</small>}
          </div>

          {/* Movie cards */}
          {reply.recommendations?.length > 0 ? (
            <motion.div
              className="ai-grid"
              variants={!reducedMotion ? staggerContainer : undefined}
              initial={!reducedMotion ? "initial" : undefined}
              animate={!reducedMotion ? "animate" : undefined}
            >
              {reply.recommendations.map(movie => (
                <motion.div
                  key={movie.movie_id}
                  variants={!reducedMotion ? staggerChild : undefined}
                >
                  <a href={href(movie)} className="ai-card">
                    <SmartPoster movie={movie} />
                    <div className="ai-card-info">
                      <h3>{movie.title}</h3>
                      <small>{movie.year} · {movie.genres?.slice(0,2).join(', ')}</small>
                      {movie.overview && <p className="ai-card-reason">{movie.overview.slice(0,120)}…</p>}
                    </div>
                  </a>
                </motion.div>
              ))}
            </motion.div>
          ) : (
            <p className="ai-empty">No matching films found for this request. Try a different genre or description.</p>
          )}
        </div>
      )}
    </SpotlightCard></section>
  );
}

// ==================== Preloader ====================
function Preloader({loading, error, onRetry}) {
  const reducedMotion = useReducedMotion();

  if (!loading && !error) return null;

  if (error) {
    return (
      <div className="preloader-error">
        <p>Could not load the film catalog.</p>
        <button onClick={onRetry}>Retry</button>
      </div>
    );
  }

  if (reducedMotion) {
    return (
      <div className="preloader" role="status" aria-label="Loading CineSense">
        <div className="preloader-simple">Loading CineSense…</div>
      </div>
    );
  }

  return (
    <div className="preloader" role="status" aria-label="Loading CineSense">
      <div className="film-frame">
        <div className="film-strip" />
        <motion.div
          className="brass-line"
          initial={{scaleX:0}}
          animate={{scaleX:1}}
          transition={{duration:1.8, ease:"easeInOut", repeat:Infinity, repeatType:"loop"}}
        />
        <motion.div
          className="preloader-brand"
          initial={{opacity:0}}
          animate={{opacity:1}}
          transition={{delay:0.6, duration:0.6}}
        >
          Cine<span>Sense</span>
        </motion.div>
      </div>
    </div>
  );
}

// ==================== Skeleton loaders ====================
function PosterSkeleton() {
  return <div className="poster skeleton"><div className="skeleton-shimmer" /></div>;
}
function CardSkeleton({count=6}) {
  return (
    <div className="shelf">
      {Array.from({length:count}, (_,i) => (
        <div key={i}>
          <PosterSkeleton />
          <div className="skeleton-text" style={{width:'60%',height:12,marginTop:9}} />
          <div className="skeleton-text" style={{width:'80%',height:18,marginTop:6}} />
        </div>
      ))}
    </div>
  );
}

function DetailSkeleton() {
  return (
    <div className="detail-skeleton">
      <div className="skeleton" style={{width:210,aspectRatio:'2/3'}} />
      <div>
        <div className="skeleton-text" style={{width:'40%',height:14}} />
        <div className="skeleton-text" style={{width:'80%',height:24,marginTop:12}} />
        <div className="skeleton-text" style={{width:'100%',height:14,marginTop:16}} />
        <div className="skeleton-text" style={{width:'90%',height:14,marginTop:6}} />
        <div className="skeleton-text" style={{width:'70%',height:14,marginTop:6}} />
      </div>
    </div>
  );
}

// ==================== About The Film Section ====================
function AboutFilm({movie, movieId}) {
  const details = useResource(`/catalog/${encodeURIComponent(movieId)}/details`);
  const d = details.data;

  if (!d && !details.error) return <DetailSkeleton />;
  if (details.error) return <div><p>{movie.overview || 'Synopsis unavailable.'}</p><p role="status">Extended details are temporarily unavailable.</p></div>;
  if (!d) return null;

  const hasOrigTitle = d.original_title && d.original_title !== d.title;
  const formatMoney = n => n ? `$${(n/1_000_000).toFixed(1)}M` : null;
  const budget = formatMoney(d.budget);
  const revenue = formatMoney(d.revenue);

  return (
    <div className="about-film">
      {/* Tagline */}
      {d.tagline && <blockquote className="tagline">"{d.tagline}"</blockquote>}

      {/* Full synopsis */}
      <div className="synopsis">
        <h3>Synopsis</h3>
        <p>{d.overview || movie.overview || 'Synopsis unavailable.'}</p>
      </div>

      {/* Metadata grid */}
      <div className="meta-grid">
        {/* Release info */}
        <div className="meta-group">
          <h4>Release</h4>
          {d.release_date && <MetaRow label="Release Date" value={formatDate(d.release_date)} />}
          {d.status && d.status !== 'Released' && <MetaRow label="Status" value={d.status} />}
          {d.runtime && <MetaRow label="Runtime" value={`${d.runtime} min`} />}
          {d.genres?.length > 0 && <MetaRow label="Genres" value={d.genres.join(', ')} />}
        </div>

        {/* Ratings */}
        <div className="meta-group">
          <h4>Ratings</h4>
          {d.vote_average > 0 && (
            <div className="rating-display">
              <span className="rating-score">★ {d.vote_average.toFixed(1)}</span>
              <span className="rating-max"> / 10</span>
              {d.vote_count && <small className="vote-count">{d.vote_count.toLocaleString()} votes</small>}
            </div>
          )}
          {d.certification && (
            <MetaRow
              label="Age Rating"
              value={`${d.certification}${d.certification_country ? ` (${d.certification_country})` : ''}`}
            />
          )}
        </div>

        {/* Language & title */}
        <div className="meta-group">
          <h4>Details</h4>
          {hasOrigTitle && <MetaRow label="Original Title" value={d.original_title} />}
          {d.original_language && <MetaRow label="Language" value={d.original_language.toUpperCase()} />}
          {budget && <MetaRow label="Budget" value={budget} />}
          {revenue && <MetaRow label="Revenue" value={revenue} />}
        </div>

        {/* Links */}
        <div className="meta-group">
          <h4>Links</h4>
          {d.imdb_id && (
            <a href={`https://www.imdb.com/title/${d.imdb_id}/`} target="_blank" rel="noopener noreferrer" className="ext-link">
              IMDb ↗
            </a>
          )}
          {d.homepage && (
            <a href={d.homepage} target="_blank" rel="noopener noreferrer" className="ext-link">
              Official site ↗
            </a>
          )}
        </div>
      </div>

      {/* Directors */}
      {d.directors?.length > 0 && (
        <div className="credits-section">
          <h3>Directed by</h3>
          <div className="credits-row">
            {d.directors.map(p => (
              <div key={p.name} className="credit-person">
                {p.profile ? <img src={p.profile} alt={p.name} loading="lazy" /> : <div className="credit-avatar">{p.name[0]}</div>}
                <span>{p.name}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Writers */}
      {d.writers?.length > 0 && (
        <div className="credits-section">
          <h3>Written by</h3>
          <div className="credits-names">
            {d.writers.map(w => <span key={w.name+w.job}>{w.name}{w.job ? ` (${w.job})` : ''}</span>).reduce((a, b) => [a, ' · ', b])}
          </div>
        </div>
      )}

      {/* Cast */}
      {d.cast?.length > 0 && (
        <div className="credits-section">
          <h3>Cast</h3>
          <div className="cast-grid">
            {d.cast.map(c => (
              <div key={c.name+c.character} className="cast-card">
                {c.profile ? <img src={c.profile} alt={c.name} loading="lazy" /> : <div className="credit-avatar">{c.name[0]}</div>}
                <div>
                  <strong>{c.name}</strong>
                  {c.character && <small>{c.character}</small>}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Production */}
      {(d.production_companies?.length > 0 || d.production_countries?.length > 0) && (
        <div className="credits-section">
          <h3>Production</h3>
          {d.production_companies?.length > 0 && (
            <div className="production-companies">
              {d.production_companies.map(c => (
                <div key={c.name} className="prod-company">
                  {c.logo && <img src={c.logo} alt={c.name} loading="lazy" />}
                  <span>{c.name}</span>
                </div>
              ))}
            </div>
          )}
          {d.production_countries?.length > 0 && (
            <MetaRow label="Countries" value={d.production_countries.join(', ')} />
          )}
        </div>
      )}
    </div>
  );
}

function MetaRow({label, value}) {
  if (!value) return null;
  return <div className="meta-row"><span className="meta-label">{label}</span><span className="meta-value">{value}</span></div>;
}

function formatDate(dateStr) {
  if (!dateStr) return null;
  try {
    return new Date(dateStr + 'T00:00:00').toLocaleDateString('en-US', {year:'numeric', month:'long', day:'numeric'});
  } catch {
    return dateStr;
  }
}

// ==================== Home ====================
function Home({catalog}) {
  const picks = {data:catalog};
  const first = picks.data?.[0];
  const art = useResource(first ? '/catalog/' + first.movie_id + '/media' : null);
  const reducedMotion = useReducedMotion();

  return (
    <>
      {first && <Hero movie={first} media={art.data} home />}
      <Search />
      <section>
        <h2>The screening room</h2>
        {picks.data ? <Shelf movies={picks.data} stagger /> : (
          picks.error ? <p>{picks.error}</p> : <CardSkeleton />
        )}
      </section>
      <Assistant />
    </>
  );
}

// ==================== Detail Page ====================
function Detail({id}) {
  const base = '/catalog/' + encodeURIComponent(id);
  const film = useResource(base);
  const artwork = useResource(base + '/media');
  const recs = useResource(base + '/recommendations?session_id=' + sid);
  const prefs = useResource('/preferences/' + sid);
  const [saved, setSaved] = useState([]);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState('');
  const dialog = useRef(null);
  const movie = film.data;
  const reducedMotion = useReducedMotion();

  async function save(interaction) {
    setBusy(true);
    try {
      await post('/preferences', {session_id: sid, movie_id: id, title: movie.title, interaction});
      setSaved(old => [...old, interaction]);
      setNotice(interaction === 'like' ? 'Added to your likes.' : 'Marked seen. Hidden from future recommendations.');
    } catch (err) {
      setNotice(err.message);
    } finally {
      setBusy(false);
    }
  }

  if (!movie) {
    if (film.status === 404) return <NotFound />;
    return (
      <section role="status">
        {film.error ? (
          <>{film.error}　<a href="/">Browse films</a></>
        ) : (
          <DetailSkeleton />
        )}
      </section>
    );
  }

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={id}
        initial={reducedMotion ? undefined : {opacity:0}}
        animate={{opacity:1}}
        exit={reducedMotion ? undefined : {opacity:0}}
        transition={{duration:0.3}}
      >
        <Hero movie={movie} media={artwork.data} />

        {/* About the Film section */}
        <section className="story">
          <SmartPoster movie={{...movie, poster_url: artwork.data?.poster_url || movie.poster_url}} />
          <div>
            <small>ABOUT THE FILM</small>
            <h2>{movie.title}</h2>
            <AboutFilm movie={movie} movieId={id} />

            {/* Controls */}
            <div className="controls">
              <button onClick={() => dialog.current.showModal()}>Watch trailer</button>
              {['like','watched'].map(k => {
                const active = saved.includes(k) || prefs.data?.some(p => p.movie_id === id && p.interaction === k);
                return (
                  <button key={k} aria-pressed={!!active} disabled={busy || active} onClick={() => save(k)}>
                    {k === 'like' ? (active ? '♥ Liked' : '♡ Like') : (active ? '✓ Seen' : 'Mark seen')}
                  </button>
                );
              })}
            </div>
            <p role="status">{notice || prefs.error}</p>
          </div>
        </section>

        {/* More like this */}
        <section>
          <h2>More like {movie.title}</h2>
          {recs.data?.length ? <Shelf movies={recs.data} stagger /> : (
            <p>{recs.error || (recs.data ? 'No suggestions yet.' : 'Finding similar films…')}</p>
          )}
        </section>

        <Trailer dialog={dialog} movie={movie} artwork={artwork} />
      </motion.div>
    </AnimatePresence>
  );
}

// ==================== Trailer ====================
function Trailer({dialog, movie, artwork}) {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    const observer = new MutationObserver(() => setOpen(dialog.current?.open));
    observer.observe(dialog.current, {attributes:true, attributeFilter:['open']});
    return () => observer.disconnect();
  }, [dialog]);
  return (
    <dialog ref={dialog} aria-label={movie.title + ' trailer'}>
      <button onClick={() => dialog.current.close()}>Close</button>
      <h2>{movie.title}</h2>
      {open && (artwork.data?.trailer_url
        ? <iframe title={movie.title + ' trailer'} src={artwork.data.trailer_url} allowFullScreen />
        : <p>{artwork.data || artwork.error ? 'Trailer unavailable.' : 'Loading trailer…'}</p>
      )}
    </dialog>
  );
}

// ==================== App ====================
function NotFound() {
  return <section className="not-found">
    <div className="missing-frame" aria-hidden="true">404</div>
    <small>THE SCREEN IS EMPTY</small>
    <h1>This scene is missing.</h1>
    <p>The page or film you requested could not be found.</p>
    <Magnet padding={30} maxOffset={5}><a className="cta" href="/">Back to discovery</a></Magnet>
  </section>;
}

function App() {
  const [path, setPath] = useState(location.pathname);
  const [initialLoading, setInitialLoading] = useState(true);
  const [loadError, setLoadError] = useState(false);
  const reducedMotion = useReducedMotion();
  const [catalog, setCatalog] = useState([]);

  // Initial catalog probe
  useEffect(() => {
    request('/discovery')
      .then(data => { setCatalog(data); setInitialLoading(false); })
      .catch(() => setLoadError(true));
  }, []);

  function retryLoad() {
    setLoadError(false);
    setInitialLoading(true);
    request('/discovery')
      .then(data => { setCatalog(data); setInitialLoading(false); })
      .catch(() => setLoadError(true));
  }

  useEffect(() => {
    history.scrollRestoration = 'manual';
    const pop = () => setPath(location.pathname);
    window.addEventListener('popstate', pop);
    return () => window.removeEventListener('popstate', pop);
  }, []);

  useLayoutEffect(() => {
    window.scrollTo(0, path === '/' ? history.state?.y || 0 : 0);
  }, [path]);

  function navigate(e) {
    const a = e.target.closest('a[href]');
    if (!a || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || a.origin !== location.origin) return;
    e.preventDefault();
    history.replaceState({...history.state, y: scrollY}, '');
    history.pushState({}, '', a.href);
    setPath(location.pathname);
  }

  if (initialLoading || loadError) {
    return (
      <main>
        <Preloader loading={initialLoading && !loadError} error={loadError} onRetry={retryLoad} />
      </main>
    );
  }

  return (
    <main onClick={navigate}>
      <nav>
        <a className="brand" href="/">Cine<span>Sense</span></a>
        <a href="/">Discover films</a>
      </nav>
      <AnimatePresence mode="wait">
        <motion.div
          key={path === '/' ? 'home' : path}
          initial={reducedMotion ? undefined : {opacity: 0}}
          animate={{opacity: 1}}
          exit={reducedMotion ? undefined : {opacity: 0}}
          transition={{duration: 0.25}}
        >
          {path === '/' ? <Home catalog={catalog}/> : (
            /^\/movie\/[\w-]+$/.test(path)
              ? <Detail key={path} id={path.slice(7)} />
              : <NotFound />
          )}
        </motion.div>
      </AnimatePresence>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
