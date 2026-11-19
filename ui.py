"""Cinematic Streamlit frontend for CineSense."""
import os
import uuid
import httpx
import streamlit as st

API = os.getenv("CINESENSE_API_URL", "http://127.0.0.1:8000")
st.set_page_config("CineSense", "🎬", layout="wide")
st.markdown("""<style>
.stApp{background:radial-gradient(circle at 20% 0,#35161d 0,#090a10 45%,#050506 100%);color:#f6f3f4}.block-container{max-width:1320px}.hero{padding:2.4rem;margin-bottom:1.5rem;border:1px solid #ffffff1e;border-radius:24px;background:linear-gradient(110deg,#df405722,#10111933)}.hero h1{font-size:clamp(3rem,8vw,6rem);letter-spacing:-.08em;line-height:.85;margin:0}.hero p{color:#d1c7cb}.score{color:#ffcf70;font-weight:bold}div.stButton>button{border-radius:9px;background:#df4057;color:white;border:1px solid #ffffff44}div.stButton>button:hover{background:#ff5b70;transform:translateY(-2px)}[data-testid="stVerticalBlockBorderWrapper"]{background:linear-gradient(145deg,#30303dbb,#13141bdd);border-color:#ffffff1c;transition:.22s}[data-testid="stVerticalBlockBorderWrapper"]:hover{transform:translateY(-6px);border-color:#ee5060;box-shadow:0 16px 42px #0009}</style>""", unsafe_allow_html=True)
if "sid" not in st.session_state: st.session_state.sid = str(uuid.uuid4())
if "selected" not in st.session_state: st.session_state.selected = ""

def call(method, path, **kwargs):
    try:
        r = httpx.request(method, API + path, timeout=50, **kwargs); r.raise_for_status(); return r.json()
    except httpx.HTTPError as error:
        st.error(f"Backend unavailable: {error}. Run `uvicorn main:app --reload`."); return None

def save(title, interaction):
    if call("POST", "/preferences", json={"session_id":st.session_state.sid,"title":title,"interaction":interaction}): st.toast("Saved locally")

def show(movie, key):
    # Widgets cannot be placed inside an HTML markdown div; a real container keeps cards clickable.
    with st.container(border=True):
        st.markdown(f"### {movie['title']}")
        st.markdown(f"<span class='score'>★ {movie['vote_average']:.1f}</span> · {' · '.join(movie['genres'][:3]) or 'Movie'}", unsafe_allow_html=True)
        st.caption((movie.get("overview") or "No overview available.")[:170] + "…")
        if movie.get("scores", {}).get("final") is not None: st.caption(f"Hybrid match: {movie['scores']['final']:.0%}")
        a,b,c = st.columns([2,1,1])
        if a.button("Recommend", key="r"+key, use_container_width=True): st.session_state.selected=movie["title"]; st.rerun()
        if b.button("♥", key="l"+key, use_container_width=True): save(movie["title"], "like")
        if c.button("Seen", key="s"+key, use_container_width=True): save(movie["title"], "watched")

st.markdown("<section class='hero'><p>LOCAL AI MOVIE DISCOVERY</p><h1>CineSense</h1><p>Your taste, your machine: hybrid recommendations + local Gemma assistant.</p></section>", unsafe_allow_html=True)
health = call("GET", "/health")
if health:
    if health["ollama_online"]: st.success(f"Local AI ready: {health['ollama_model']}")
    else: st.info("Recommendations ready hain. AI explanation ke liye `ollama run gemma2:2b` start karo.")
query = st.text_input("Find a movie", placeholder="Search Avatar, Toy Story, Interstellar…")
quick_a, quick_b, quick_c = st.columns(3)
if quick_a.button("Try Avatar", use_container_width=True): st.session_state.selected="Avatar"; st.rerun()
if quick_b.button("Try Toy Story", use_container_width=True): st.session_state.selected="Toy Story"; st.rerun()
if quick_c.button("Sci-Fi suggestions", use_container_width=True): st.session_state.selected="Interstellar"; st.rerun()
if query:
    items=call("GET", "/movies/search", params={"q":query})
    if items:
        st.subheader("Search results"); cols=st.columns(4)
        for n,movie in enumerate(items):
            with cols[n%4]: show(movie, "q"+str(n))
if st.session_state.selected:
    title=st.session_state.selected; st.subheader(f"Because you chose {title}")
    media = call("GET", f"/media/{title}")
    if media and media.get("trailer_url"):
        st.components.v1.iframe(media["trailer_url"], height=420, scrolling=False)
        st.caption("Muted trailer preview — use the player controls to unmute.")
    elif media and media.get("poster_url"):
        st.image(media["poster_url"], width=300)
    else:
        st.caption("Add `TMDB_API_KEY` to `.env` to enable optional poster and muted trailer previews.")
    items=call("GET", f"/recommendations/{title}", params={"session_id":st.session_state.sid})
    if items:
        cols=st.columns(4)
        for n,movie in enumerate(items):
            with cols[n%4]: show(movie, "m"+str(n))
st.divider(); st.subheader("Ask CineSense AI")
message=st.chat_input("Hinglish works: Mujhe fast-paced sci-fi chahiye, romance kam ho")
if message:
    with st.chat_message("user"): st.write(message)
    answer=call("POST", "/assistant/chat", json={"session_id":st.session_state.sid,"message":message})
    if answer:
        with st.chat_message("assistant"): st.write(answer["reply"])
        cols=st.columns(3)
        for n,movie in enumerate(answer["recommendations"][:3]):
            with cols[n]: show(movie, "c"+str(n))
