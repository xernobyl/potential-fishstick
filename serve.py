#!/usr/bin/env python3
"""Static server for local development, with caching OFF.

`python3 -m http.server` sends no Cache-Control, so the browser applies heuristic freshness and
will happily serve a stale `.wgsl` or `.js` after an edit. That failure is genuinely confusing,
because the shader compiler then reports an error against source that no longer exists on disk,
and a hard reload does not always clear it. Every response here says no-store.

    python3 serve.py [port]
"""

import functools
import http.server
import os
import sys


class NoCache(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, must-revalidate')
        super().end_headers()

    def log_message(self, fmt, *args):
        pass                      # one line per shader include is just noise


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    root = os.path.dirname(os.path.abspath(__file__))
    handler = functools.partial(NoCache, directory=root)
    print(f'http://localhost:{port}  (serving {root}, caching disabled)')
    try:
        http.server.ThreadingHTTPServer(('127.0.0.1', port), handler).serve_forever()
    except KeyboardInterrupt:
        pass
