"""Flask server for real-time point tracking with WebSocket support."""

from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image
import io
import base64
from threading import Timer, Lock, Thread
from queue import Queue, Empty
import webbrowser
import os
import time
from copilot import CoTrackerRealTimePredictor

# Server settings
HOST, PORT = '127.0.0.1', 5000
current_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize Flask and Socket
app = Flask(__name__, template_folder=current_dir, static_folder=current_dir)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

class TrackerManager:
    """Manages tracking requests queue and processing."""
    def __init__(self, tracker):
        self.tracker = tracker
        self.request_queue = Queue(maxsize=2)  # Keep only latest 2 requests
        self.processing_lock = Lock()
        Thread(target=self._process_queue, daemon=True).start()

    def add_request(self, frame, points, request_id):
        """Add new request, dropping oldest if queue is full."""
        while self.request_queue.full():
            try:
                self.request_queue.get_nowait()
            except Empty:
                break
        self.request_queue.put_nowait({
            'frame': frame, 'points': points,
            'request_id': request_id, 'timestamp': time.time()
        })

    def _process_queue(self):
        """Process requests from queue with rate limiting."""
        while True:
            try:
                request = self.request_queue.get()
                if time.time() - request['timestamp'] > 1.0:
                    continue

                with self.processing_lock:
                    points, visibility = self.tracker(
                        request['frame'],
                        request['points']
                    )
                    socketio.emit('tracking_update', {
                        'updatedPoints': format_points(points, visibility),
                        'requestId': request['request_id']
                    })
                # time.sleep(0.033)  # Rate limiting
            except Exception as e:
                print(f"Processing error: {e}")

def process_image(image_data):
    """Convert base64 image to tensor."""
    img_bytes = base64.b64decode(image_data.split(',')[1])
    img_array = np.array(Image.open(io.BytesIO(img_bytes)))
    return torch.from_numpy(img_array.astype(np.float32)).permute(2,0,1)[None].cuda()

def process_points(points, device):
    """Convert points to tensor if available."""
    if not points:
        return None
    return torch.tensor([[p['x'], p['y']] for p in points])[None].to(device)

def format_points(points, visibility):
    """Format tracking results for response."""
    if points is None:
        return []
    points = points.squeeze(0)
    visibility = visibility.squeeze(0).squeeze(-1)
    return [{
        'x': float(point[0].item()),
        'y': float(point[1].item()),
        'isVisible': bool(vis.item())
    } for point, vis in zip(points, visibility)]

# Route handlers
@app.route('/')
def index():
    return send_from_directory(current_dir, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(current_dir, filename)

# Socket handlers
@socketio.on('reset_tracker')
def handle_reset():
    try:
        tracker_manager.tracker.reset()
        emit('reset_complete')
    except Exception as e:
        emit('reset_error', {'message': str(e)})

@socketio.on('update_tracking')
def handle_tracking(data):
    try:
        frame = process_image(data['frame'])
        points = process_points(data.get('points'), frame.device)
        tracker_manager.add_request(frame, points, data.get('requestId'))
    except Exception as e:
        emit('tracking_error', {'message': str(e)})

if __name__ == '__main__':
    print("\n=== Real-time Tracking Server ===")
    
    # Initialize tracker
    print("Initializing tracker...")
    tracker = CoTrackerRealTimePredictor().cuda()
    tracker.warm_up()
    tracker_manager = TrackerManager(tracker)
    print("Tracker ready!")
    
    print(f"\nServer URL: http://{HOST}:{PORT}")
    Timer(1.5, lambda: webbrowser.open(f'http://{HOST}:{PORT}')).start()
    
    socketio.run(app, host=HOST, port=PORT, debug=True, use_reloader=False)