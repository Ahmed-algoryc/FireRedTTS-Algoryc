#!/usr/bin/env python3
"""
Simple HTTP Server to serve the HTML interface
"""

import http.server
import socketserver
import webbrowser
import os
import threading
import time

def start_http_server():
    """Start HTTP server to serve the HTML interface"""
    PORT = 8080
    
    # Change to the directory containing the HTML file
    os.chdir('/root/Projects/FireRedTTS2')
    
    Handler = http.server.SimpleHTTPRequestHandler
    
    with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
        print(f"🌐 HTTP Server running at http://0.0.0.0:{PORT}")
        print(f"📄 Open: http://localhost:{PORT}/voice_chat.html")
        print("🔌 Make sure the WebSocket server is running on port 8765")
        httpd.serve_forever()

if __name__ == "__main__":
    start_http_server()

