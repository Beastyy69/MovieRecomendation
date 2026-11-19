"""CineSense FastAPI backend: TF-IDF recommendations + optional Gemini intent/explanations."""
import asyncio
import ast
import hashlib
import json
import logging
import os
import pickle
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('cinesense')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '').strip()
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash').strip()
TMDB_API_KEY = os.getenv('TMDB_API_KEY', '').strip()
DB_PATH = Path(os.getenv('DB_PATH', str(ROOT / 'user_preferences.db')))
CORS_ORIGINS = [x.strip().rstrip('/') for x in os.getenv('CORS_ORIGINS', 'https://cinesense-6s8i.onrender.com,http://localhost:5173,http://127.0.0.1:5173').split(',') if x.strip()]
app = FastAPI(title='CineSense AI', version='3.0')
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
AI_GATE = asyncio.Semaphore(3)
df: Optional[pd.DataFrame] = None
matrix: Any = None
vectorizer: Any = None
title_index: dict[str, int] = {}
catalog: dict[str, int] = {}
movie_ids: list[str] = []
all_genres: list[set[str]] = []
media_cache: dict[str, tuple[float, dict]] = {}
poster_cache: dict[str, tuple[float, Optional[str]]] = {}
details_cache: dict[str, tuple[float, dict]] = {}

KNOWN_GENRES = ('Science Fiction', 'TV Movie', 'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery', 'Romance', 'Thriller', 'War', 'Western')
ALIASES = {'sci-fi':'Science Fiction','scifi':'Science Fiction','science fiction':'Science Fiction','science-fiction':'Science Fiction','space':'Science Fiction','comedy':'Comedy','funny':'Comedy','hasna':'Comedy','horror':'Horror','scary':'Horror','darawna':'Horror','darr':'Horror','romance':'Romance','romantic':'Romance','pyaar':'Romance','love':'Romance','thriller':'Thriller','suspense':'Thriller','mystery':'Mystery','action':'Action','fight':'Action','adventure':'Adventure','fantasy':'Fantasy','jadoo':'Fantasy','family':'Family','parivaar':'Family','animation':'Animation','cartoon':'Animation','crime':'Crime','war':'War','jung':'War','drama':'Drama','emotional':'Drama','documentary':'Documentary','history':'History','music':'Music','western':'Western'}
MOODS = {'dark':['dark','bleak','grim','disturbing','dystopian','sinister','psychological','noir','unsettling','post apocalyptic','post-apocalyptic'], 'funny':['funny','humorous','hilarious','lighthearted','comedy'], 'romantic':['romance','romantic','love','relationship'], 'scary':['scary','horror','terrifying','haunted','ghost'], 'mind-bending':['mind bending','mind-bending','twist','time travel','reality','paradox','mystery'], 'feel-good':['feel good','feel-good','uplifting','heartwarming','wholesome','happy']}
STOPWORDS = {'mujhe','kuch','chahiye','batao','suggest','karo','please','movies','movie','film','films','type','jaisi','jaisa','aisi','aisa','recommend','me','some','want','i','a','the','to','of','for','with','de','do','hai','hain'}

class Preference(BaseModel):
    movie_id: Optional[str] = None
    session_id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1)
    interaction: str = Field(pattern='^(like|dislike|watched|rating)$')
    rating: Optional[float] = Field(default=None, ge=1, le=5)

class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=1000)

def normalise(v: str) -> str:
    return ' '.join(str(v).lower().split())

def genres(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [g for g in KNOWN_GENRES if any(normalise(str(v)) == normalise(g) for v in value)]
    if value is None or not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            if parsed and isinstance(parsed[0], dict):
                return genres([x.get('name','') for x in parsed])
            return genres(parsed)
    except (ValueError, SyntaxError, TypeError):
        pass
    return [g for g in KNOWN_GENRES if re.search(r'(?<!\w)' + re.escape(g) + r'(?!\w)', value, re.I)]

def detect_genres(text: str) -> set[str]:
    return {genre for alias, genre in ALIASES.items() if re.search(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', text, re.I)}

def detect_moods(text: str) -> set[str]:
    return {m for m, terms in MOODS.items() if any(re.search(r'(?<!\w)' + re.escape(t) + r'(?!\w)', text, re.I) for t in terms)}

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as con:
        con.execute('''CREATE TABLE IF NOT EXISTS preferences (session_id TEXT, title TEXT, interaction TEXT, rating REAL, updated_at TEXT DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (session_id,title,interaction))''')
        con.commit()

def load_model():
    global df, matrix, vectorizer, title_index, all_genres
    required = ['movies.pkl','tfidf_matrix.pkl','tfidf_vectorizer.pkl','indices.pkl','movies_metadata.csv']
    missing = [name for name in required if not (ROOT/name).exists()]
    if missing:
        raise RuntimeError('Missing model files: ' + ', '.join(missing))
    with open(ROOT/'movies.pkl','rb') as f: df = pickle.load(f)
    with open(ROOT/'tfidf_matrix.pkl','rb') as f: matrix = pickle.load(f)
    with open(ROOT/'tfidf_vectorizer.pkl','rb') as f: vectorizer = pickle.load(f)
    with open(ROOT/'indices.pkl','rb') as f: saved = pickle.load(f)
    df = df.reset_index(drop=True)
    title_index = {normalise(str(t)): int(i) for t,i in saved.items() if isinstance(i,(int,np.integer)) and 0 <= int(i) < len(df)}
    raw = pd.read_csv(ROOT/'movies_metadata.csv',dtype=str,usecols=['id','title','overview','poster_path','runtime','release_date']).fillna('')
    raw = raw.drop_duplicates(['title','overview']).set_index(['title','overview'])
    extra = raw.reindex(pd.MultiIndex.from_frame(df[['title','overview']].fillna('').astype(str))).reset_index(drop=True)
    for col in ('id','poster_path','runtime','release_date'):
        if col not in df: df[col] = extra[col].fillna('')
    catalog.clear(); movie_ids.clear()
    for i,row in df.iterrows():
        key = str(row.get('id',''))
        if not key.isdigit() or key in catalog:
            key = 'film-' + hashlib.sha256(f"{key}|{row.get('title')}|{row.get('overview')}|{row.get('release_date')}".encode()).hexdigest()[:24]
        while key in catalog: key += '-2'
        catalog[key] = i
        movie_ids.append(key)
    all_genres = [set(genres(v)) for v in df['genres']]
    if matrix.shape[0] != len(df): raise RuntimeError('TF-IDF matrix and movie catalog have different row counts')
    log.info('Loaded %s films', len(df))

@app.on_event('startup')
def startup():
    init_db(); load_model()

def ready():
    if df is None or matrix is None or vectorizer is None:
        raise HTTPException(503, 'Recommendation model is not loaded')

def numeric_column(name: str) -> pd.Series:
    value = df.loc[:,name] if name in df.columns else pd.Series(0,index=df.index)
    if isinstance(value,pd.DataFrame): value = value.iloc[:,0]
    return pd.to_numeric(value,errors='coerce').fillna(0)

def clean(v: Any) -> str:
    if v is None or pd.isna(v): return ''
    s = str(v)
    return '' if s.lower() in ('nan','none','nat') else s

def card(i: int, scores: Optional[dict] = None) -> dict[str,Any]:
    row = df.iloc[int(i)]
    poster = clean(row.get('poster_path',''))
    mid = movie_ids[i]
    return {'movie_id':mid,'year':clean(row.get('release_date',''))[:4], 'runtime':clean(row.get('runtime','')), 'title':clean(row.get('title','')), 'overview':clean(row.get('overview','')), 'genres':sorted(all_genres[i]), 'vote_average':float(numeric_column('vote_average').iloc[i]), 'popularity':float(numeric_column('popularity').iloc[i]), 'scores':scores or {}, 'poster_url':'https://image.tmdb.org/t/p/w500'+poster if poster.startswith('/') else None, 'tmdb_id':clean(row.get('id','')) if clean(row.get('id','')).isdigit() else None,'trailer_url':None}

def prefs(session_id: str) -> pd.DataFrame:
    with closing(sqlite3.connect(DB_PATH)) as con:
        return pd.read_sql_query('SELECT title,interaction,rating FROM preferences WHERE session_id=?',con,params=(session_id,))

def personal_score(session_id: str) -> tuple[np.ndarray,set[int]]:
    score = np.zeros(matrix.shape[0]); watched=set(); liked=[]; disliked=[]; weights=[]
    for item in prefs(session_id).itertuples(index=False):
        i = catalog.get(item.title[9:]) if item.title.startswith('@catalog:') else title_index.get(normalise(item.title))
        if i is None: continue
        if item.interaction == 'watched': watched.add(i)
        elif item.interaction == 'dislike' or (item.interaction == 'rating' and (item.rating or 3) <= 2): disliked.append(i)
        elif item.interaction == 'like' or (item.interaction == 'rating' and (item.rating or 0) >= 4):
            liked.append(i); weights.append(1 if item.interaction == 'like' else (item.rating-3)/2)
    if liked: score += np.average(cosine_similarity(matrix,matrix[liked]),axis=1,weights=weights)
    if disliked: score -= .6*cosine_similarity(matrix,matrix[disliked]).max(axis=1)
    return np.clip(score,-1,1),watched

def rank(content: np.ndarray, session_id: str, limit: int, source: Optional[int] = None, genre_filter: Optional[set[str]] = None, moods: Optional[set[str]] = None) -> list[dict[str,Any]]:
    content = np.asarray(content,dtype=float).ravel()
    user,watched = personal_score(session_id)
    popularity = np.log1p(np.maximum(numeric_column('popularity').to_numpy(float),0))
    popularity /= max(float(popularity.max()),1)
    target_genres = all_genres[source] if source is not None else (genre_filter or set())
    overlap = np.array([len(g & target_genres)/len(g | target_genres) if g and target_genres else 0 for g in all_genres])
    if source is not None:
        final = .80*content+.15*overlap+.04*user+.01*popularity
    else:
        # For natural-language queries, genre and plot matter more than global popularity.
        final = .62*content+.32*overlap+.05*user+.01*popularity
    mood_bonus = np.zeros(len(df))
    if moods:
        for i, overview in enumerate(df['overview'].fillna('').astype(str)):
            text = overview.lower()
            matched = sum(any(re.search(r'(?<!\w)'+re.escape(word)+r'(?!\w)',text) for word in MOODS[mood]) for mood in moods)
            mood_bonus[i] = matched/max(len(moods),1)
        final += .10*mood_bonus
    # Discourage entries with no plot and no useful metadata.
    overview_lengths = df['overview'].fillna('').astype(str).str.len().to_numpy()
    final = np.where(overview_lengths < 25, final*.60, final)
    output=[]
    for j in np.argsort(final)[::-1]:
        i=int(j)
        if i == source or i in watched: continue
        if genre_filter and not (all_genres[i] & genre_filter): continue
        if source is not None and (content[i] < .035 or (target_genres and not all_genres[i] & target_genres)): continue
        if source is None and not genre_filter and content[i] <= .001: continue
        output.append(card(i, {'content':round(float(content[i]),4),'preference':round(float(user[i]),4),'popularity':round(float(popularity[i]),4),'final':round(float(final[i]),4)}))
        if len(output)>=limit: break
    return output

def by_title(title: str, session_id: str, limit: int):
    ready(); i=title_index.get(normalise(title))
    if i is None: raise HTTPException(404,f"Movie '{title}' was not found")
    return rank(cosine_similarity(matrix[i],matrix).ravel(),session_id,limit,source=i)

def resolve(movie_id: str) -> int:
    if movie_id not in catalog: raise HTTPException(404,'Film not found in this catalog')
    return catalog[movie_id]

async def gemini_generate(prompt: str, json_mode: bool = False) -> Optional[str]:
    if not GEMINI_API_KEY: return None
    url=f'https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent'
    payload={'contents':[{'parts':[{'text':prompt}]}],'generationConfig':{'temperature':0.2,'maxOutputTokens':700}}
    if json_mode: payload['generationConfig']['responseMimeType']='application/json'
    try:
        async with AI_GATE:
            async with httpx.AsyncClient(timeout=25) as client:
                response=await client.post(url,headers={'x-goog-api-key':GEMINI_API_KEY},json=payload)
                response.raise_for_status()
        parts=response.json().get('candidates',[{}])[0].get('content',{}).get('parts',[])
        return ''.join(p.get('text','') for p in parts).strip() or None
    except (httpx.HTTPError,ValueError,IndexError,KeyError) as exc:
        log.warning('Gemini unavailable: %s',str(exc)[:300])
        return None

async def understand_query(message: str) -> tuple[set[str],set[str],str]:
    detected=detect_genres(message); moods=detect_moods(message)
    words=[w for w in re.findall(r'[\w-]+',message.lower()) if w not in STOPWORDS]
    search=' '.join(words) or message
    if not GEMINI_API_KEY: return detected,moods,search
    prompt=('Extract movie-search intent. Return ONLY a JSON object with keys genres (array of exact names from '+str(list(KNOWN_GENRES))+'), moods (array from '+str(list(MOODS))+'), search_terms (short English plot/theme keywords). Do not suggest movie titles. Request: '+message)
    raw=await gemini_generate(prompt,json_mode=True)
    if not raw: return detected,moods,search
    try:
        obj=json.loads(raw)
        extracted={x for x in obj.get('genres',[]) if x in KNOWN_GENRES}
        mood_extracted={x for x in obj.get('moods',[]) if x in MOODS}
        terms=str(obj.get('search_terms','')).strip()[:180]
        return extracted or detected,mood_extracted or moods,terms or search
    except (ValueError,TypeError,AttributeError):
        return detected,moods,search

async def explain(message: str, movies: list[dict]) -> Optional[str]:
    if not movies: return None
    context=[{'title':m['title'],'genres':m['genres'],'overview':m['overview'][:250]} for m in movies[:6]]
    prompt=('You are CineSense. Answer in Hinglish if the user writes Hinglish. Discuss ONLY films in the JSON catalog; do not invent films or claim poor matches are perfect. Explain 2-4 relevant choices in 2-4 concise sentences. Plain text, no markdown. User: '+message+'\nCatalog: '+json.dumps(context,ensure_ascii=False))
    return await gemini_generate(prompt)

def fallback_explanation(movies: list[dict]) -> str:
    if not movies: return 'Is request ke liye catalog mein koi reliable match nahi mila. Try a different genre or mood.'
    return 'Catalog picks: '+', '.join(f"{m['title']} ({', '.join(m['genres'][:2])})" for m in movies[:3])+'. These are based on available genre and plot matches.'

@app.get('/healthz')
def healthz():
    ready(); return {'status':'ok'}

@app.get('/health')
def health():
    return {'status':'ok','model_loaded':df is not None,'ai_provider':'gemini','gemini_configured':bool(GEMINI_API_KEY),'gemini_model':GEMINI_MODEL}

@app.get('/movies/search')
def search(q: str=Query(min_length=1),limit: int=Query(8,ge=1,le=20)):
    ready(); ids=df.index[df['title'].astype(str).str.contains(q,case=False,regex=False,na=False)][:limit]
    return [card(int(i)) for i in ids]

@app.get('/recommendations/{title}')
def recommendations(title: str,session_id: str='guest',limit: int=Query(8,ge=1,le=20)):
    return by_title(title,session_id,limit)

@app.get('/discovery')
def discovery():
    ready(); return [card(title_index[t]) for t in ('avatar','thor: ragnarok','interstellar','toy story','spirited away','inception') if t in title_index]

@app.get('/catalog/{movie_id}')
def film_lookup(movie_id: str):
    return card(resolve(movie_id))

@app.get('/catalog/{movie_id}/recommendations')
def similar(movie_id: str,session_id: str='guest'):
    i=resolve(movie_id)
    return rank(cosine_similarity(matrix[i],matrix).ravel(),session_id,12,source=i)

async def tmdb_get(path: str,params: Optional[dict]=None) -> Optional[dict]:
    if not TMDB_API_KEY: return None
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response=await client.get('https://api.themoviedb.org/3/'+path.lstrip('/'),params={'api_key':TMDB_API_KEY,**(params or {})})
            response.raise_for_status()
            return response.json()
    except (httpx.HTTPError,ValueError): return None

async def tmdb_movie(title: str,index: Optional[int]) -> Optional[dict]:
    if index is not None:
        mid=clean(df.iloc[index].get('id',''))
        if mid.isdigit():
            result=await tmdb_get('movie/'+mid)
            if result: return result
    params={'query':title}
    if index is not None:
        year=clean(df.iloc[index].get('release_date',''))[:4]
        if year.isdigit(): params['year']=year
    data=await tmdb_get('search/movie',params)
    return next(iter(data.get('results',[])),None) if data else None

async def fetch_media(title: str,index: Optional[int]):
    if not TMDB_API_KEY: return {'poster_url':None,'trailer_url':None}
    key=movie_ids[index] if index is not None else title
    cached=media_cache.get(key)
    if cached and time.monotonic()-cached[0]<3600: return cached[1]
    movie=await tmdb_movie(title,index)
    if not movie: return {'poster_url':None,'trailer_url':None}
    videos=await tmdb_get(f"movie/{movie['id']}/videos")
    trailer=next((v for v in (videos or {}).get('results',[]) if v.get('site')=='YouTube' and v.get('type')=='Trailer'),None)
    poster=movie.get('poster_path'); backdrop=movie.get('backdrop_path')
    payload={'poster_url':'https://image.tmdb.org/t/p/w500'+poster if poster else None,'backdrop_url':'https://image.tmdb.org/t/p/w1280'+backdrop if backdrop else None,'year':str(movie.get('release_date',''))[:4],'trailer_url':f"https://www.youtube.com/embed/{trailer['key']}?autoplay=1&mute=1&controls=1" if trailer else None}
    media_cache[key]=(time.monotonic(),payload)
    return payload

@app.get('/media/{title}')
async def media(title: str):
    return await fetch_media(title,title_index.get(normalise(title)))

@app.get('/catalog/{movie_id}/media')
async def artwork(movie_id: str):
    i=resolve(movie_id)
    return await fetch_media(str(df.iloc[i]['title']),i)

@app.get('/catalog/{movie_id}/poster')
async def poster_lookup(movie_id: str):
    i=resolve(movie_id)
    cached=poster_cache.get(movie_id)
    if cached and time.monotonic()-cached[0]<3600: return {'poster_url':cached[1],'movie_id':movie_id}
    poster=clean(df.iloc[i].get('poster_path',''))
    url='https://image.tmdb.org/t/p/w500'+poster if poster.startswith('/') else None
    if url:
        try:
            async with httpx.AsyncClient(timeout=4) as client:
                r=await client.head(url)
                if r.status_code!=200: url=None
        except httpx.HTTPError: url=None
    if not url:
        movie=await tmdb_movie(str(df.iloc[i]['title']),i)
        path=movie.get('poster_path') if movie else None
        url='https://image.tmdb.org/t/p/w500'+path if path else None
    poster_cache[movie_id]=(time.monotonic(),url)
    return {'poster_url':url,'movie_id':movie_id}

@app.get('/catalog/{movie_id}/details')
async def movie_details(movie_id: str):
    i=resolve(movie_id)
    cached=details_cache.get(movie_id)
    if cached and time.monotonic()-cached[0]<3600: return cached[1]
    row=df.iloc[i]
    base={'movie_id':movie_id,'title':clean(row.get('title')),'overview':clean(row.get('overview')),'genres':sorted(all_genres[i]),'year':clean(row.get('release_date'))[:4],'runtime':None,'vote_average':float(numeric_column('vote_average').iloc[i]),'tagline':None,'original_title':None,'original_language':None,'status':None,'release_date':clean(row.get('release_date')) or None,'budget':None,'revenue':None,'vote_count':None,'imdb_id':None,'homepage':None,'production_companies':[],'production_countries':[],'directors':[],'writers':[],'cast':[],'certification':None,'certification_country':None,'poster_url':None,'backdrop_url':None}
    try: base['runtime']=int(float(clean(row.get('runtime')))) or None
    except ValueError: pass
    movie=await tmdb_movie(base['title'],i)
    if movie:
        mid=movie['id']
        detail,credits,releases=await asyncio.gather(tmdb_get(f'movie/{mid}'),tmdb_get(f'movie/{mid}/credits'),tmdb_get(f'movie/{mid}/release_dates'))
        d=detail or movie
        for key in ('tagline','original_title','original_language','status','release_date','budget','revenue','vote_count','imdb_id','homepage','runtime'):
            if d.get(key): base[key]=d[key]
        if d.get('overview'): base['overview']=d['overview']
        if d.get('vote_average'): base['vote_average']=d['vote_average']
        if d.get('genres'): base['genres']=[g['name'] for g in d['genres']]
        base['year']=str(base.get('release_date') or '')[:4]
        if d.get('poster_path'): base['poster_url']='https://image.tmdb.org/t/p/w500'+d['poster_path']
        if d.get('backdrop_path'): base['backdrop_url']='https://image.tmdb.org/t/p/w1280'+d['backdrop_path']
        base['production_companies']=[{'name':c['name'],'logo':'https://image.tmdb.org/t/p/w200'+c['logo_path'] if c.get('logo_path') else None,'country':c.get('origin_country')} for c in d.get('production_companies',[])]
        base['production_countries']=[c.get('name','') for c in d.get('production_countries',[])]
        cr=credits or {}
        base['directors']=[{'name':c['name'],'profile':'https://image.tmdb.org/t/p/w185'+c['profile_path'] if c.get('profile_path') else None} for c in cr.get('crew',[]) if c.get('job')=='Director']
        base['writers']=[{'name':c['name'],'job':c.get('job','')} for c in cr.get('crew',[]) if c.get('department')=='Writing'][:6]
        base['cast']=[{'name':c['name'],'character':c.get('character',''),'profile':'https://image.tmdb.org/t/p/w185'+c['profile_path'] if c.get('profile_path') else None} for c in cr.get('cast',[])[:12]]
        for country in ('US','GB','IN',None):
            for item in (releases or {}).get('results',[]):
                if country and item.get('iso_3166_1')!=country: continue
                cert=next((r.get('certification','').strip() for r in item.get('release_dates',[]) if r.get('certification','').strip()),None)
                if cert:
                    base['certification']=cert; base['certification_country']=item.get('iso_3166_1'); break
            if base['certification']: break
    details_cache[movie_id]=(time.monotonic(),base)
    return base

@app.post('/preferences')
def save_preference(value: Preference):
    ready()
    if normalise(value.title) not in title_index: raise HTTPException(404,'Movie not found')
    storage=value.title
    if value.movie_id:
        i=resolve(value.movie_id)
        if str(df.iloc[i]['title'])!=value.title: raise HTTPException(422,'Movie ID and title differ')
        if title_index[normalise(value.title)]!=i: storage='@catalog:'+value.movie_id
    if value.interaction=='rating' and value.rating is None: raise HTTPException(422,'Rating is required')
    with closing(sqlite3.connect(DB_PATH)) as con:
        con.execute('''INSERT INTO preferences(session_id,title,interaction,rating) VALUES(?,?,?,?) ON CONFLICT(session_id,title,interaction) DO UPDATE SET rating=excluded.rating,updated_at=CURRENT_TIMESTAMP''',(value.session_id,storage,value.interaction,value.rating))
        con.commit()
    return {'status':'saved'}

@app.get('/preferences/{session_id}')
def read_preferences(session_id: str):
    records=prefs(session_id).astype(object).where(lambda x:x.notna(),None).to_dict('records')
    for item in records:
        t=item['title']; i=catalog.get(t[9:]) if t.startswith('@catalog:') else title_index.get(normalise(t))
        if i is not None: item.update(movie_id=movie_ids[i],title=str(df.iloc[i]['title']))
    return records

@app.post('/assistant/chat')
async def chat(value: ChatRequest):
    ready()
    detected,moods,search_terms=await understand_query(value.message)
    query=normalise(value.message)
    # Match full film titles only, not common short fragments inside unrelated sentences.
    matched=next((t for t in sorted(title_index,key=len,reverse=True) if len(t)>=5 and re.search(r'(?<!\w)'+re.escape(t)+r'(?!\w)',query)),None)
    if matched and not detected:
        movies=by_title(matched,value.session_id,6)
    else:
        search_terms=search_terms or value.message
        content=cosine_similarity(vectorizer.transform([search_terms]),matrix).ravel()
        movies=rank(content,value.session_id,6,genre_filter=detected or None,moods=moods)
    ai_response=await explain(value.message,movies)
    reply=ai_response or fallback_explanation(movies)
    return {'reply':reply,'recommendations':movies,'ai_available':bool(ai_response),'detected_genres':sorted(detected)}
