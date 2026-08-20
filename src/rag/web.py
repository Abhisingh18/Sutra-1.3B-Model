"""Live web retrieval, for questions a static index cannot answer.

The Wikipedia index is a snapshot -- 20231101.en -- so anything after that date
is simply not in it. This swaps the corpus for a search engine and keeps
everything else the same: it returns the SAME (score, text, source) tuples as
src.rag.retrieve.Retriever, so build_prompt and the server's source display
work against it unchanged.

    from src.rag.web import WebRetriever
    w = WebRetriever()                       # picks a provider from the env
    for score, text, source in w.search("who won the last world cup"):
        ...

No relevance gate here, unlike the Wikipedia retriever. That gate exists
because a dense index returns its nearest neighbour whether or not it is about
the question; a search engine already decides relevance, and a query with no
good answer comes back empty rather than confidently wrong.

Set ONE of these:

    TAVILY_API_KEY   tavily.com          1,000 searches/month free
    BRAVE_API_KEY    brave.com/search/api  2,000/month free

DuckDuckGo's HTML endpoint is deliberately not used: it answers scripted
requests with an anomaly page rather than results.
"""

import json
import os
import urllib.parse
import urllib.request

TIMEOUT = 20


class WebRetriever:
    def __init__(self, provider=None):
        self.tavily = os.environ.get("TAVILY_API_KEY", "").strip()
        self.brave = os.environ.get("BRAVE_API_KEY", "").strip()

        self.provider = provider or (
            "tavily" if self.tavily else "brave" if self.brave else None
        )
        if self.provider == "tavily" and not self.tavily:
            raise RuntimeError("TAVILY_API_KEY is not set")
        if self.provider == "brave" and not self.brave:
            raise RuntimeError("BRAVE_API_KEY is not set")
        if self.provider is None:
            raise RuntimeError(
                "no search key found -- set TAVILY_API_KEY or BRAVE_API_KEY")

    @property
    def available(self):
        return self.provider is not None

    def search(self, query, k=4, min_chars=120):
        try:
            hits = (self._tavily(query, k) if self.provider == "tavily"
                    else self._brave(query, k))
        except Exception as e:
            # A search outage must not take the chat down; answering without
            # context is worse than answering with it, but far better than an
            # error page.
            print(f"web search failed: {type(e).__name__}: {e}", flush=True)
            return []

        # Snippets shorter than a sentence or two carry no usable fact and
        # only crowd the context window.
        return [h for h in hits if len(h[1]) >= min_chars]

    # ---- providers ------------------------------------------------------

    def _tavily(self, query, k):
        body = json.dumps({
            "api_key": self.tavily,
            "query": query,
            "max_results": k,
            "search_depth": "basic",
            # The generator writes the answer; an engine-written one would just
            # be a better model's text passed off as this model's.
            "include_answer": False,
        }).encode()
        req = urllib.request.Request(
            "https://api.tavily.com/search", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = json.load(r)

        out = []
        for item in data.get("results", []):
            text = (item.get("content") or "").strip()
            name = self._host(item.get("url", "")) or item.get("title", "web")
            out.append((float(item.get("score") or 0.0), text, name))
        return out

    def _brave(self, query, k):
        url = ("https://api.search.brave.com/res/v1/web/search?q="
               + urllib.parse.quote(query) + f"&count={k}")
        req = urllib.request.Request(url, headers={
            "Accept": "application/json",
            "X-Subscription-Token": self.brave,
        })
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = json.load(r)

        out = []
        for i, item in enumerate(data.get("web", {}).get("results", [])[:k]):
            text = (item.get("description") or "").strip()
            name = self._host(item.get("url", "")) or item.get("title", "web")
            # Brave returns no score; rank stands in for one so the caller can
            # order and threshold the same way it does for the dense index.
            out.append((1.0 - i * 0.1, text, name))
        return out

    @staticmethod
    def _host(url):
        try:
            return urllib.parse.urlparse(url).netloc.replace("www.", "")
        except Exception:
            return ""


def available_provider():
    """Which provider a key exists for, or None. Never raises."""
    if os.environ.get("TAVILY_API_KEY", "").strip():
        return "tavily"
    if os.environ.get("BRAVE_API_KEY", "").strip():
        return "brave"
    return None
