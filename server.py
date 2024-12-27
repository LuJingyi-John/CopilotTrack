"""Flask server for real-time point tracking with WebSocket support."""

from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image
import io
import base64
from threading import Timer
import webbrowser
import os

from copilot import CoTrackerRealTimePredictor

# Server configuration
HOST = '127.0.0.1'  # Use '0.0.0.0' for remote access
PORT = 5000
SERVER_URL = f'http://{HOST}:{PORT}'

# Initialize Flask app with root directory as template/static folder
current_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=current_dir, static_folder=current_dir)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize tracker
print("Initializing tracker...")
tracker = CoTrackerRealTimePredictor().cuda()
tracker.warm_up()
print("Tracker ready!")

def process_image(image_data):
    """Convert base64 image to CUDA tensor (1,3,H,W)."""
    img_bytes = base64.b64decode(image_data.split(',')[1])
    img_array = np.array(Image.open(io.BytesIO(img_bytes)))
    return torch.from_numpy(img_array.astype(np.float32)).permute(2,0,1)[None].cuda()

def process_points(points, device):
    """Convert points list to tensor (1,N,2) if available."""
    if not points:
        return None
    return torch.tensor([[p['x'], p['y']] for p in points])[None].to(device)

def format_points(points, visibility):
    """Format tracking points with visibility for response."""
    if points is None:
        return []
    
    points = points.squeeze(0)
    result = []
    visibility = visibility.squeeze(0).squeeze(-1)
    for point, vis in zip(points, visibility):
        result.append({
            'x': float(point[0].item()),
            'y': float(point[1].item()),
            'isVisible': bool(vis.item())
        })
    return result

# Route handlers
@app.route('/')
def index():
    """Serve index.html from root directory."""
    return send_from_directory(current_dir, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files from root directory."""
    return send_from_directory(current_dir, filename)

# WebSocket handlers
@socketio.on('reset_tracker')
def handle_reset():
    """Reset tracker states."""
    try:
        tracker.reset()
        emit('reset_complete')
    except Exception as e:
        emit('reset_error', {'message': str(e)})

@socketio.on('update_tracking')
def handle_tracking(data):
    """Process frame and update point tracking."""
    try:
        frame = process_image(data['frame'])
        new_queries = process_points(data.get('points'), frame.device)
        updated_points, updated_visibility = tracker(frame, new_queries)
        
        emit('tracking_update', {
            'updatedPoints': format_points(updated_points, updated_visibility),
            'requestId': data.get('requestId')
        })
    except Exception as e:
        print(f"Error in tracking: {e}")
        emit('tracking_error', {'message': str(e)})

def check_files():
    """Check if required files exist."""
    required_files = ['index.html']
    missing = [f for f in required_files 
              if not os.path.exists(os.path.join(current_dir, f))]
    if missing:
        print("\nMissing required files:", *missing, sep='\n- ')
        return False
    return True

if __name__ == '__main__':
    print("\n=== Real-time Tracking Server ===")
    
    # Verify file structure
    if not check_files():
        print("\nPlease ensure all required files are present")
        exit(1)
    
    print(f"\nServer ready!")
    print(f"➜ URL: {SERVER_URL}")
    print("\nNotes:")
    print("1. Browser will open automatically")
    print("2. Press Ctrl+C to stop server")
    
    # Auto open browser after server starts
    Timer(1.5, lambda: webbrowser.open(SERVER_URL)).start()
    
    # Start server
    socketio.run(
        app, 
        host=HOST, 
        port=PORT, 
        debug=True,
        use_reloader=False  # Disable reloader for CUDA compatibility
    )